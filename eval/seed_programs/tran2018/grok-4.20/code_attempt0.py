import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['parameters']['n']
    R = instance['parameters']['R']
    p = instance['parameters']['p']
    od_pairs = instance['od_pairs']
    num_od = len(od_pairs)
    nodes = {node['id']: node for node in instance['nodes']}

    # Precompute for each OD the list of positions where refueling can cover each segment
    # For each OD, we build a list of "requirements": for each critical segment, the set of nodes that can cover it
    od_data = []
    for od_idx, od in enumerate(od_pairs):
        path = od['shortest_path']  # nodes from O to D
        flow = od['flow']
        path_dist = od['path_distance']
        
        # Build cumulative distance from origin (for outbound)
        cum_dist_out = [0.0]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            # Find arc distance
            dist = None
            for arc in instance['arcs']:
                if (arc['from'] == u and arc['to'] == v) or (arc['from'] == v and arc['to'] == u):
                    dist = arc['distance']
                    break
            cum_dist_out.append(cum_dist_out[-1] + dist)
        
        # Return path is reverse, cumulative from D (but since round trip starts from O with full tank, return starts with whatever left at D)
        # But according to problem: depart from origin with full tank only once. So on return, fuel left when arriving at D is R - path_dist if path_dist <= R else impossible.
        # But problem says: "directed arcs whose cumulative distance from the origin does not exceed the vehicle's driving range are automatically covered by the initial full tank"
        # The round-trip is O->D then D->O. The "origin" here is the start of the round trip, i.e. O.
        # So cumulative distance is along the entire round trip path.
        
        full_round_cum = []
        # Outbound: O to D
        for d in cum_dist_out:
            full_round_cum.append(d)
        # Inbound: from D back to O, cumulative continues
        for i in range(len(path)-1, 0, -1):
            u = path[i]
            v = path[i-1]
            dist = None
            for arc in instance['arcs']:
                if (arc['from'] == u and arc['to'] == v) or (arc['from'] == v and arc['to'] == u):
                    dist = arc['distance']
                    break
            full_round_cum.append(full_round_cum[-1] + dist)
        
        total_round_dist = 2 * path_dist
        # Now, the round trip has 2*(len(path)-1) directed arcs.
        # But we need to identify for each directed arc whether it needs a station and which stations can cover it.
        
        # A vehicle refuels at stations along the way. The coverage of a station is that from the refuel point, it can go up to R distance.
        # Since drivers refuel optimally with full knowledge.
        # This is similar to the "refueling station location for round trips" where we need to cover the path such that no gap > R.
        
        # But the problem gives specific definition:
        # "directed arcs whose cumulative distance from the origin does not exceed the vehicle's driving range are automatically covered by the initial full tank and require no open station."
        # "For each remaining directed arc on the round trip, there must be at least one open station among the set of candidate nodes that can service that arc."
        # "A candidate node can service a directed arc if a vehicle that refuels at that node has enough fuel to traverse the arc without running dry, given the vehicle's range and the distances between nodes along the path."
        
        # So it seems they model it as: the initial tank covers up to distance R from O along the round-trip path.
        # Then for any arc beyond that, it must be "serviced" by some station, meaning there is a station where the vehicle could have refueled last before that arc, such that from that station's location along the path, the distance to the arc is <=R.
        
        # To make it concrete, this is equivalent to requiring that along the entire round-trip path (of length 2*path_dist), the maximum distance between consecutive refueling points (including the initial fill at O and any stations on the path) is <= R.
        # But the problem says "vehicles are assumed to depart from the origin with a full tank only once at the start of the round trip (there is no refueling at the destination before the return leg)".
        # So O is always a "virtual" full tank start, but we cannot assume refueling at O unless we place a station there (but since it starts full, it's like initial fill is always possible at O even without station?).
        # The initial tank is always full at start, so first R is always covered.
        # For the return, if path_dist > R, then on the way back we may need refueling on the outbound or at some point.
        
        # To operationalize "a candidate node can service a directed arc":
        # A station at a node on the path can "cover" all arcs that are within R distance after it along the round trip path.
        
        # So to cover the entire path, every point on the path (or every arc) must be within R of either the origin (initial fill) or some open station along the path (in the order of travel).
        # Since it's a path, it's like the stations divide the path into segments <=R.
        # But because it's round trip on same path, and start only at O.
        
        # The standard way for such problems is to find all possible "refueling stops" positions on the path.
        # Given that this is an optimization problem to solve with MIP (given Gurobi is required), we will formulate it as a MIP.
        
        # Let's define for each OD pair, the positions where a station can be useful, i.e. the nodes on its round trip path.
        # Since paths are given, and nodes are few (we assume n is small because single core and gurobi).
        
        # First, we need to know what n is typically. Since not specified, we assume n<=100 or so for MIP to work.
        
        # The decision is which p nodes to open as stations.
        # For each OD, it is covered if there exists a set of opened stations that "cover" all its "critical" segments.
        # But to model "can complete round trip", we can model for each OD the maximum distance between refuel points.
        
        # A good way is to precompute for each OD, the list of nodes on its path that are "possible refueling locations".
        # The round trip path is a sequence of nodes: path_out = path, path_back = path[::-1][1:] so full_node_sequence = path + path[-2::-1] or something.
        
        # Let's define the positions along the round trip.
        # Let’s say the full path is a line of length 2*path_dist, starting at O with fuel R.
        # A station at a node corresponds to a position(s) on this line (since node can be visited twice, outbound and inbound).
        # But since it's the same node, opening a station there allows refueling both times it is visited.
        
        # This is getting complicated. Given time, let's look for a standard formulation for "refueling station location to maximize covered flow with p stations".
        
        # A common model is the "Flow Refueling Location Model" (FRLM) by Kuby et al, but for round trips it's a bit different.
        # But the problem seems to have a specific definition.
        
        # The problem says: "A given OD pair's traffic flow is considered "refueled" (and its flow volume counts toward the total) only if every directed arc on the round-trip path can be traversed by a vehicle that refuels at one of the open stations. Specifically, directed arcs whose cumulative distance from the origin does not exceed the vehicle's driving range are automatically covered by the initial full tank and require no open station. For each remaining directed arc on the round trip, there must be at least one open station among the set of candidate nodes that can service that arc. A candidate node can service a directed arc if a vehicle that refuels at that node has enough fuel to traverse the arc without running dry, given the vehicle's range and the distances between nodes along the path."
        
        # So, it seems they consider that for arcs within first R from start, covered.
        # For arcs beyond, each such arc must have at least one station that can "service" it.
        # And "service" means that if you refuel at that candidate node, you can reach the arc and traverse it with the fuel.
        # This implies that the station must be before the arc in the path, and distance from station to the end of the arc <=R.
        
        # Importantly, since multiple stations can be used, it's not that one station covers all remaining, but that the stations together allow the vehicle to refuel along the way so that every arc is reachable from the last refuel point (which could be a station or initial).
        
        # If it was independent per arc, that would be too loose because it wouldn't enforce that you can reach the station itself.
        # The "drivers refuel optimally" suggests that the model should ensure there exists a refueling policy that allows completing the trip.
        
        # The standard way to model this is to discretize the possible refueling stops to the nodes on the path.
        # Assume we can only refuel at the selected stations that are on the OD's path.
        # Then, the condition is that from O, you can reach the first station or if no need, the entire if <=R, but with the round trip consideration.
        
        # To make it solvable, since we have to use MIP and Gurobi, and time limit, let's assume n is small.
        # Let's first see if we can get the number of nodes.
        
        # I will formulate it as an Integer Program.
        # Decision: x_i = 1 if we build station at node i, sum x = p, binary.
        # For each OD k, we have a binary y_k = 1 if this OD is fully covered by the stations.
        # Maximize sum flow_k * y_k
        # The hard part is to link y_k to the x's.
        
        # To do that, we need to know under what condition on the x's the y_k can be 1.
        # That is, when the selected stations allow the round trip.
        
        # To model if a path is feasible with given stations.
        # Since the path is a line for round trip: we can think of the full path as a sequence of nodes with cumulative distances from the starting point.
        # Let S be the sequence of positions (with possible duplicates if node visited twice).
        # But since station at node is available at both visits.
        
        # Let's create for each OD a list of possible "refuel points": the nodes on the path (unique or with positions).
        # A good approximation / exact way for path coverage is:
        # Consider all the locations where the vehicle could run out of fuel.
        # But a standard exact way for "can the vehicle make the round trip with range R with stations at selected locations" is to simulate the trip:
        # Start with fuel = R at O.
        # For each segment along the round trip, subtract the distance.
        # If fuel would go negative before reaching a point, fail.
        # But if you pass by an open station, you can refuel to full before continuing.
        # Since "refuel optimally", the driver will refuel at a station if it helps.
        # Since it's a path without loops, the optimal is to refuel at every station you visit (to full).
        # So we can simulate: go along the path, keep track of current fuel.
        # When you arrive at a node with station, you can fill up to R before leaving it.
        # So current_fuel = R when leaving a station node.
        # For the return, same path.
        # Note that if you place station at O, it doesn't add because you start full anyway. Station at D allows you to fill at D for the return trip.
        
        # This is the key.
        # To check if a set of stations covers an OD: 
        # Take the full round trip node sequence: outbound path then inbound path reversed without duplicating D.
        # full_path = path + path[-2::-1]   # e.g. if path = [1,2,3], full = [1,2,3,2,1]
        # Then distances between consecutive.
        # Start at first node (O), fuel = R.
        # For i in range(1, len(full_path)):
        #   dist_to_next = distance between full_path[i-1] and full_path[i]
        #   if current_fuel < dist_to_next: cannot make it.
        #   current_fuel -= dist_to_next
        #   if full_path[i] has station or (i==0):  # fill up
        #       current_fuel = R
        # But at start we start with full, so before leaving O, fuel=R.
        # If station at a node, when we arrive at it, we can refuel before leaving it.
        # So the code is:
        # current_fuel = R
        # for i in range(1, len(full_path)):
        #     dist = distance(full_path[i-1], full_path[i])
        #     if current_fuel < dist:
        #         return False
        #     current_fuel -= dist
        #     # Upon arrival at full_path[i], if it has a station, we refuel
        #     if full_path[i] in open_stations:
        #         current_fuel = R
        # return True
        
        # Note that if O has a station it doesn't change anything because we start with R anyway.
        # If D has a station, when we arrive at D, we refuel to R before starting the return.
        # Which matches "no refueling at the destination before the return leg" ONLY if no station at destination.
        # But if we place a station at destination, then we can refuel there.
        # The problem says "there is no refueling at the destination before the return leg" but I think it means there is no automatic refuel, only if we place a station.
        # I think the above logic is correct.
        
        # Also, for arcs beyond initial R, it is covered by the stations.
        
        # This logic matches the description.
        
        # Since n is probably small, but to use in MIP, we cannot directly use this logic easily because it's "or" conditions along the path.
        # One way is to use the concept of "covering" by having auxiliary variables for whether we "use" a station as the last refuel for certain segments.
        
        # But if the number of OD pairs is large but n small, or vice versa.
        # The schema has "m" as number of OD nodes, "num_od_pairs".
        # To make the MIP work, we can use a formulation where for each OD, we introduce the concept of which stations are used for refueling.
        
        # Since we have time limit and single core, and we need good solution, perhaps we can use a greedy or heuristic, but since Gurobi is suggested, let's try a MIP.
        
        # First, let's assume n is small (<=30), p small, then we can even enumerate but better MIP.
        # The y_k can be linked if we precompute all subsets of stations that cover the OD, but that is 2^n which is bad.
        
        # A better way: for a fixed OD, the path is fixed, the full round trip is a fixed sequence of nodes with fixed distances between them.
        # Let’s say we have the sequence of visited nodes: let seq = full_path as above, let d[i] = distance from seq[i-1] to seq[i]
        # To model if with the selected stations the trip is feasible, we can use auxiliary continuous variables for the fuel level when arriving at each point in the sequence.
        # But since fuel is reset to R at stations if open.
        # But because whether a station is open is variable, it's nonlinear unless we use big-M or indicators.
        
        # Since Gurobi supports indicators, or we can use big M.
        # But with many OD pairs, if num_od_pairs is 100 and path length 20, then number of constraints is large but maybe acceptable.
        
        # Let's try to formulate.
        # For each OD k, to have y_k =1 only if the path is feasible.
        # We introduce for each position i in the sequence (after each node), a variable f_i = fuel level when leaving node i.
        # But it's a bit tricky.
        
        # Since the refuel happens only at stations or start, the key points are the locations of possible stations on the path.
        # Let the candidate refuel nodes for an OD be the unique nodes on its path (except perhaps O).
        # But to check feasibility is equivalent to: the distance between any two consecutive "refuel points" (including O and D if station, and the stations selected) along the *round trip* must be <=R.
        # The selected stations on the path, plus the origin as initial, we consider all refueling events in the order they appear in the round trip.
        # The positions where we can refuel are at the nodes if station is placed.
        # So if we take all nodes in the full_path that have station (plus the starting O), then the distance between consecutive refuel opportunities in the order of the trip must be <= R.
        # Yes.
        # So if we extract the list of positions (cumulative distance) where there is a station, then add 0 at start, and check that no two consecutive have diff >R, and also from last to end if needed but since we only need to reach the end, the last segment to the end must be feasible but since we don't need fuel at end, it's the gaps between refuels <=R.
        # Since we refuel at a station and then can go R from there.
        # So if refuel positions (cumulative dist from start) are 0 = p0 < p1 < p2 < ... < pk, then we need p_{i+1} - p_i <= R for all i, and total_length - p_k <=R ? No, if the last refuel allows reaching the end.
        # Since after last refuel, as long as from last refuel position the end is within R, yes.
        # But if the end is the return to O, and if we have station at O it would be at the end too but since we don't need to refuel at end, we only need to reach it.
        # So yes, if we include position 0 as always refuel (initial), and if station at last node (O on return) it would allow but since it's the end it's not necessary.
        # The simulation I wrote earlier with the for loop is the most straightforward.
        
        # To linearize it in MIP is a bit complex but possible by having fuel variables and using implications.
        
        # Since single CPU core and time limit, if n is not too large we can use Gurobi to solve the MIP with y_k and use no direct constraint but perhaps use a heuristic.
        
        # Let's think of a different approach.
        # This seems like a maximal covering location problem but with specific coverage.
        # First, I need to precompute for each OD, which subsets of nodes "cover" it, but that is exponential.
        # If the paths are short, the number of relevant nodes per path is small (<=10), then 2^10 = 1024, we can precompute all minimal covering sets but still for MIP it's better to use dynamic constraints or lazy.
        # But that is complicated.
        
        # Since the problem asks for an efficient program and to get as close to optimal as possible, perhaps a greedy algorithm: iteratively add the station that covers the most additional flow.
        # This is the greedy for set cover like.
        # To do it, we need to be able to evaluate for a set of stations what is the total flow covered.
        # So first I need a function that, given a set of station nodes, computes which OD pairs are covered and the total flow.
        
        # To do that, I need to implement the "is_covered" function for one OD given stations.
        # Let's implement the simulation as I said earlier.
        
        # First, I need distance between any two consecutive nodes. Since we have the path, we can precompute the edge distances on the path.
        
def is_od_covered(od, stations, R, instance):
    path = od['shortest_path']
    if not path:
        return True
    # full round trip node sequence
    full_path = path[:]
    for i in range(len(path)-2, -1, -1):  # add return without duplicating destination
        full_path.append(path[i])
    # now full_path starts at O ends at O
    
    # We need distances between consecutive in full_path
    # To avoid searching arcs every time, let's build a dict of distances
    arc_dict = {}
    for arc in instance['arcs']:
        a, b, d = arc['from'], arc['to'], arc['distance']
        arc_dict[(a,b)] = d
        arc_dict[(b,a)] = d
    
    current_fuel = R
    for i in range(1, len(full_path)):
        u = full_path[i-1]
        v = full_path[i]
        dist = arc_dict[(u, v)]
        if current_fuel < dist - 1e-6:  # allowing floating point tolerance
            return False
        current_fuel -= dist
        # if we arrive at a station, refuel to full
        if v in stations:
            current_fuel = R
    return True

# Now, with this function, we can solve the problem using a MIP if n is small, or using heuristic.
# Let's see what n is. The parameters have "n", let's assume we can use MIP with y_k but link using callback or something but too complex.
# Since p is given, if p and n small we can use brute force but not.
# A good compromise is to use Gurobi to solve a set covering like but we need the relation.
# Since the coverage is complex, we can use a heuristic: greedy adding one station at a time that improves the objective the most.
# We can run it until time limit, logging improvements.
# This is efficient.
# To improve quality, we can use random greedy or do local search swapping stations.
# But for starters, let's implement greedy.
# Since we have Gurobi, perhaps we can model it as quadratic or no.
# Another way: the number of possible "relevant" station locations is probably all nodes, but many nodes may not be on any OD path.
# But let's count how many unique nodes are used in all shortest paths.
# But to use MIP, one common way for this kind of "path covering with fuel" is to find all possible arcs or segments that need covering and what stations cover each segment.
# According to the problem statement, it seems they intend for us to consider independent "arc covering" but with the cumulative.
# But the last sentence "Drivers have full knowledge of station locations along their path and refuel optimally to complete their journey." suggests the simulation approach is correct.
# So I will use that.
# To get optimal, since single core, we can use a MIP formulation with fuel variables per OD per position but it would be heavy if paths are long.
# Let's see if we can do it.
# For each OD, the number of nodes in full_path is 2* (len(path)-1).
# If average |path| = 10, then ~20 positions, if 100 OD pairs, then 2000 variables per OD would be too many? No.
# Let's try to formulate a MIP.
# For each OD k:
# We have y_k binary, if covered.
# We have for each position i=0 to L-1 (L = len(full_path)), a continuous var fuel_i : fuel when leaving the i-th node in sequence.
# But to force the logic only when y_k=1.
# It's a bit tricky but we can use:
# We can ignore y first and maximize sum flow*y and use indicator constraints or bigM to enforce the fuel logic only if y=1.
# But if y=0 we don't care about the fuel.
# One way is to not use fuel variables, but instead use the concept that the last refuel position before each point.
# A standard formulation for fuel constrained shortest path is to have discrete fuel levels but here since we only decide station locations, it's the opposite.
# Given the time, I decide to implement a greedy + local search heuristic that uses the is_covered function.
# It should be fast and give good solutions.
# We will iteratively add the station that gives the largest increase in objective until we place p stations.
# Then, we do local search: try swapping one selected with one not selected if improves.
# To use the time limit, we can run as many improvements as possible within time.
# Also log every time the objective improves.
    
    start_time = time.time()
    remaining_time = args.time_limit
    
    # Get all possible station locations. Assume all nodes can have stations.
    all_nodes = list(range(1, n+1))  # assuming nodes are 1 to n. Check instance.
    # From schema, nodes have "id", so let's collect
    node_ids = [node['id'] for node in instance['nodes']]
    # Usually 0-index or 1, but use as is.
    possible_stations = set()
    for od in od_pairs:
        for nd in od['shortest_path']:
            possible_stations.add(nd)
    # Only nodes that appear in some path can be useful. But to be safe use all.
    possible_stations = list(possible_stations)
    if len(possible_stations) == 0:
        possible_stations = node_ids
    
    # Greedy
    selected = set()
    current_covered = set()
    current_obj = 0.0
    
    def compute_covered_and_obj(selected_stations):
        covered_indices = []
        total = 0.0
        for idx, od in enumerate(od_pairs):
            if is_od_covered(od, selected_stations, R, instance):
                covered_indices.append(idx)
                total += od['flow']
        return covered_indices, total
    
    # Greedy placement
    for _ in range(p):
        best_add = None
        best_inc = -1
        best_covered = None
        best_obj = None
        for candidate in possible_stations:
            if candidate in selected:
                continue
            temp_selected = selected | {candidate}
            covered_idx, obj_val = compute_covered_and_obj(temp_selected)
            inc = obj_val - current_obj
            if inc > best_inc:
                best_inc = inc
                best_add = candidate
                best_covered = covered_idx
                best_obj = obj_val
        if best_add is None or best_inc <= 0:
            break
        selected.add(best_add)
        current_obj = best_obj
        current_covered = set(best_covered)  # use set for ease
        if logger:
            logger.log(current_obj)
        elapsed = time.time() - start_time
        if elapsed > args.time_limit * 0.95:
            break
    
    # Now local search: try to swap
    improved = True
    while improved and (time.time() - start_time < args.time_limit):
        improved = False
        current_selected_list = list(selected)
        not_selected = [nd for nd in possible_stations if nd not in selected]
        for i, in_sel in enumerate(current_selected_list):
            for out_sel in not_selected:
                temp_selected = set(current_selected_list)
                temp_selected.remove(in_sel)
                temp_selected.add(out_sel)
                _, new_obj = compute_covered_and_obj(temp_selected)
                if new_obj > current_obj + 1e-6:
                    selected = temp_selected
                    current_obj = new_obj
                    current_covered, _ = compute_covered_and_obj(selected)  # recompute indices
                    if logger:
                        logger.log(current_obj)
                    improved = True
                    break
            if improved:
                break
        # also try to add if p not reached but we have exactly p
    
    # If we have time, we can do more iterations with random perturbations but this should be good.
    # If p is large and n large, this may be slow because each evaluation is O(num_od * path_length), if num_od=100, path=20, it's 2000 operations, and if n=100, p=20 then greedy is 20*80* eval = about 1600 evals, totally fine.
    # Local search can be slow if we do full, but we can limit the number of swaps.
    # To make it faster in local search, we can randomly sample candidates instead of full loop.
    
    elapsed = time.time() - start_time
    if elapsed < args.time_limit and len(selected) < p:
        # fill with random if needed but usually we stop at p
        pass
    
    final_covered_list, final_obj = compute_covered_and_obj(selected)
    final_stations = sorted(list(selected))
    
    solution = {
        "objective_value": final_obj,
        "stations": final_stations,
        "refueled_od_pairs": sorted(final_covered_list)
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    if logger:
        logger.log(final_obj)  # final log

if __name__ == "__main__":
    main()