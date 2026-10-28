import argparse
import json
import time
import sys
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        inst = json.load(f)

    net = inst['network']
    tp = inst['time_parameters']
    demand = inst['demand']
    fleet = inst['fleet']

    T = tp['time_horizon_T']
    nodes = {n['id']: n for n in net['nodes']}
    links_list = net['links']
    
    centroids = [n['id'] for n in net['nodes'] if n['type'] == 'centroid']
    junctions = [n['id'] for n in net['nodes'] if n['type'] == 'junction']
    centroid_set = set(centroids)
    junction_set = set(junctions)

    # Build link lookup
    link_by_id = {}
    link_by_endpoints = {}
    for l in links_list:
        link_by_id[l['id']] = l
        link_by_endpoints[(l['from'], l['to'])] = l

    # Build adjacency
    outgoing = defaultdict(list)  # node -> list of links going out
    incoming = defaultdict(list)  # node -> list of links coming in
    for l in links_list:
        outgoing[l['from']].append(l)
        incoming[l['to']].append(l)

    # Identify link types
    road_links = [l for l in links_list if l['type'] == 'road']
    cc_links = [l for l in links_list if l['type'] == 'centroid_connector']
    
    # Outgoing centroid connectors: from centroid to junction
    outgoing_cc = [l for l in cc_links if l['from'] in centroid_set]
    # Incoming centroid connectors: from junction to centroid
    incoming_cc = [l for l in cc_links if l['to'] in centroid_set]

    outgoing_cc_by_centroid = defaultdict(list)
    for l in outgoing_cc:
        outgoing_cc_by_centroid[l['from']].append(l)
    
    incoming_cc_by_centroid = defaultdict(list)
    for l in incoming_cc:
        incoming_cc_by_centroid[l['to']].append(l)

    # Build demand
    # demand at (origin, destination, time) -> number of travelers
    demand_data = defaultdict(lambda: defaultdict(float))  # (o,d) -> {t: count}
    od_pairs_set = set()
    for od in demand['od_pairs']:
        o = od['origin']
        d = od['destination']
        od_pairs_set.add((o, d))
        for dep_t in od['departure_times']:
            demand_data[(o, d)][dep_t] += 1.0

    od_pairs = list(od_pairs_set)
    
    # All destinations that appear
    all_destinations = set()
    for o, d in od_pairs:
        all_destinations.add(d)
    all_destinations = list(all_destinations)
    dest_set = set(all_destinations)

    # Initial fleet distribution
    initial_parked = {}
    for c in centroids:
        initial_parked[c] = fleet['initial_distribution'].get(c, 0)

    # For turning flows, we need to enumerate valid turns:
    # A turn is (from_link, to_link) where from_link.to == to_link.from
    # We need turns at junctions (and also flows leaving/entering centroid connectors)
    
    # Let me think about the structure more carefully.
    # 
    # Links: road links between junctions, outgoing CC (centroid->junction), incoming CC (junction->centroid)
    #
    # At a junction node v:
    #   Incoming links: road links ending at v, outgoing CCs ending at v (from centroid to v)
    #   Outgoing links: road links starting at v, incoming CCs starting at v (from v to centroid)
    #
    # Turning flows y[a, b, s, t] = vehicles going from link a through node v to link b, 
    #   with destination s, at time t
    # where a ends at v and b starts at v, v is a junction
    #
    # For centroid departures: y_centroid[c, j, s, t] = vehicles departing centroid c 
    #   via outgoing CC (c->j) toward destination s at time t
    
    # Actually, let me re-read the schema more carefully.
    # y key: '{from_node}|{via_node}|{to_node}|{destination_centroid}|{time_step}'
    # This represents turning from link (from_node -> via_node) to link (via_node -> to_node)
    
    # y_centroid key: '{centroid}|{junction}|{destination_centroid}|{time_step}'
    # Vehicles departing centroid via outgoing connector to junction
    
    # N_U, N_D key: '{from_node}|{to_node}|{destination_centroid}|{time_step}'
    
    import gurobipy as gp
    from gurobipy import GRB

    start_time = time.time()
    
    m = gp.Model("SAV")
    m.Params.TimeLimit = max(1, time_limit - 10)
    m.Params.Threads = 1
    m.Params.Method = 2  # Barrier for LP
    m.Params.OutputFlag = 1

    # Determine which destinations are relevant for each link
    # To keep the model manageable, we track all destinations on all links
    # But we can be smarter: a destination s only needs to be tracked on links 
    # that could be on a path to s.
    
    # For efficiency, let's do BFS from each destination backwards to find reachable links
    # Actually, let's find which links can reach each centroid destination
    links_for_dest = {}
    nodes_reaching_dest = {}
    
    for s in all_destinations:
        # BFS backward from s
        # s is a centroid. Incoming CCs go to s from junctions.
        # We need to find all nodes that can reach s
        reachable = set()
        queue = [s]
        reachable.add(s)
        while queue:
            node = queue.pop()
            for l in incoming[node]:
                if l['from'] not in reachable:
                    reachable.add(l['from'])
                    queue.append(l['from'])
        nodes_reaching_dest[s] = reachable
        links_for_dest[s] = []
        for l in links_list:
            if l['from'] in reachable and l['to'] in reachable:
                links_for_dest[s].append(l)

    # Similarly find which destinations are relevant per link
    dests_for_link = {}
    for l in links_list:
        dests = []
        for s in all_destinations:
            if l['from'] in nodes_reaching_dest[s] and l['to'] in nodes_reaching_dest[s]:
                dests.append(s)
        dests_for_link[(l['from'], l['to'])] = dests

    # Enumerate turns at junctions
    # A turn at junction v: from link (u->v) to link (v->w)
    turns_at = defaultdict(list)  # junction -> list of (in_link, out_link)
    for v in junctions:
        in_links = incoming[v]
        out_links = outgoing[v]
        for il in in_links:
            for ol in out_links:
                turns_at[v].append((il, ol))

    # Variables
    # y[u, v, w, s, t] - turning flow from link(u,v) to link(v,w) for dest s at time t
    # We only create variables where needed
    
    print(f"Creating variables... T={T}, links={len(links_list)}, dests={len(all_destinations)}")
    sys.stdout.flush()
    
    # y variables for turns at junctions
    y_vars = {}
    for v in junctions:
        for (il, ol) in turns_at[v]:
            u = il['from']
            w = ol['to']
            fftt_il = il['free_flow_travel_time_steps']
            # Destinations relevant for both links
            dests_il = set(dests_for_link.get((u, v), []))
            dests_ol = set(dests_for_link.get((v, w), []))
            common_dests = dests_il & dests_ol
            
            # If ol is an incoming CC (v -> centroid), only the centroid's dest is valid
            if ol['type'] == 'centroid_connector' and w in centroid_set:
                common_dests = common_dests & {w}
            
            for s in common_dests:
                for t in range(fftt_il, T):
                    y_vars[(u, v, w, s, t)] = m.addVar(lb=0, name=f"y_{u}_{v}_{w}_{s}_{t}")

    # y_centroid[c, j, s, t] - vehicles departing centroid c via connector (c->j) for dest s at time t
    yc_vars = {}
    for c in centroids:
        for l in outgoing_cc_by_centroid[c]:
            j = l['to']
            for s in all_destinations:
                if j in nodes_reaching_dest.get(s, set()):
                    for t in range(T):
                        yc_vars[(c, j, s, t)] = m.addVar(lb=0, name=f"yc_{c}_{j}_{s}_{t}")

    # N_U[u, v, s, t] - upstream cumulative count on link (u,v) for dest s at time t
    # N_D[u, v, s, t] - downstream cumulative count
    NU_vars = {}
    ND_vars = {}
    for l in links_list:
        u, v = l['from'], l['to']
        dests = dests_for_link.get((u, v), [])
        if l['type'] == 'centroid_connector' and v in centroid_set:
            # Incoming CC: only destination = v
            dests = [v] if v in dest_set else []
        for s in dests:
            for t in range(T + 1):
                NU_vars[(u, v, s, t)] = m.addVar(lb=0, name=f"NU_{u}_{v}_{s}_{t}")
                ND_vars[(u, v, s, t)] = m.addVar(lb=0, name=f"ND_{u}_{v}_{s}_{t}")

    # p[c, t] - parked vehicles at centroid c at time t
    p_vars = {}
    for c in centroids:
        for t in range(T + 1):
            p_vars[(c, t)] = m.addVar(lb=0, name=f"p_{c}_{t}")

    # e[o, d, t] - travelers picked up at o for d at time t
    e_vars = {}
    for (o, d) in od_pairs:
        for t in range(T):
            e_vars[(o, d, t)] = m.addVar(lb=0, name=f"e_{o}_{d}_{t}")

    # omega[o, d, t] - waiting travelers at o for d at time t
    omega_vars = {}
    for (o, d) in od_pairs:
        for t in range(T + 1):
            omega_vars[(o, d, t)] = m.addVar(lb=0, name=f"om_{o}_{d}_{t}")

    m.update()
    print(f"Variables created: {m.NumVars}")
    sys.stdout.flush()

    # CONSTRAINTS

    # 1. Initial conditions: N_U and N_D at t=0 are 0
    for l in links_list:
        u, v = l['from'], l['to']
        dests = dests_for_link.get((u, v), [])
        if l['type'] == 'centroid_connector' and v in centroid_set:
            dests = [v] if v in dest_set else []
        for s in dests:
            m.addConstr(NU_vars[(u, v, s, 0)] == 0)
            m.addConstr(ND_vars[(u, v, s, 0)] == 0)

    # 2. N_U update for road links and outgoing CCs
    # For road links and outgoing CCs:
    # N_U[u,v,s,t+1] = N_U[u,v,s,t] + sum over w of y[u,v,w,s,t]  (total turning flow out of (u,v) for dest s at t)
    # Wait, re-reading: "the upstream cumulative count for a given destination at the next time step 
    # equals the current upstream count plus the total turning flow of vehicles with that destination 
    # from that link onto all of its successor links at the current time step"
    # 
    # Hmm, but that seems backwards. Upstream count should track entries, not exits.
    # Let me re-read more carefully.
    #
    # "For each road link and each outgoing centroid connector, the upstream cumulative count for a 
    # given destination at the next time step equals the current upstream count plus the total turning 
    # flow of vehicles with that destination from that link onto all of its successor links at the 
    # current time step."
    #
    # Wait, this says N_U[link, s, t+1] = N_U[link, s, t] + sum_successor y[link->successor, s, t]
    # That means upstream count increases when vehicles LEAVE the link?? That doesn't make sense for 
    # "upstream cumulative count"...
    #
    # Let me re-read once more: "the upstream cumulative count... equals the current upstream count 
    # plus the total turning flow of vehicles with that destination from that link onto all of its 
    # successor links"
    #
    # Hmm, actually I think the problem statement might be using a specific LTM convention.
    # Let me re-read the full paragraph:
    #
    # "For each road link and each outgoing centroid connector, the upstream cumulative count for a 
    # given destination at the next time step equals the current upstream count plus the total turning 
    # flow of vehicles with that destination from that link onto all of its successor links at the 
    # current time step."
    #
    # OK wait, I think "from that link" means from the PREDECESSOR link. Let me re-read:
    # "turning flow of vehicles with that destination from that link onto all of its successor links"
    # "that link" = the link we're computing N_U for? Or the predecessor?
    #
    # Actually, I think I need to interpret this differently. Let me look at both statements together:
    #
    # Statement 1: "For each road link and each outgoing centroid connector, the upstream cumulative 
    # count for a given destination at the next time step equals the current upstream count plus the 
    # total turning flow of vehicles with that destination from that link onto all of its successor 
    # links at the current time step."
    #
    # This says: N_U(link_a, s, t+1) = N_U(link_a, s, t) + sum_{b in successors(a)} y(a->b, s, t)
    # "from that link" = link_a, "onto successor links" = successors of link_a
    #
    # Statement 2: "For each road link and each incoming centroid connector, the downstream cumulative 
    # count for a given destination at the next time step equals the current downstream count plus the 
    # total turning flow of vehicles with that destination from all predecessor links onto that link 
    # at the current time step."
    #
    # This says: N_D(link_b, s, t+1) = N_D(link_b, s, t) + sum_{a in predecessors(b)} y(a->b, s, t)
    # "from all predecessor links" onto "that link" = link_b
    #
    # So N_U tracks outflow from the link, and N_D tracks inflow to the link.
    # This is actually the OPPOSITE of what I'd expect, but it makes sense in the LTM context
    # where you track cumulative counts at both ends.
    #
    # Actually wait - in LTM, N_U is the cumulative number of vehicles that have ENTERED the upstream 
    # end, and N_D is the cumulative number that have EXITED the downstream end. 
    # 
    # But the way turning flows work: a turning flow y from link a to link b simultaneously represents
    # vehicles exiting link a (downstream of a) and entering link b (upstream of b).
    #
    # Hmm, but the problem says:
    # N_U(a, s, t+1) = N_U(a, s, t) + sum_b y(a->b, s, t)  -- this adds outgoing turns FROM a
    # N_D(b, s, t+1) = N_D(b, s, t) + sum_a y(a->b, s, t)  -- this adds incoming turns TO b
    #
    # So N_U tracks vehicles leaving a, and N_D tracks vehicles entering b.
    # But conventionally N_U should track upstream (entry) and N_D downstream (exit).
    #
    # Hmm, maybe the naming is swapped from convention, or maybe I need to think about it differently.
    # Let me just follow the problem statement exactly.
    #
    # The sending flow constraint says:
    # "the total turning flow of vehicles with that destination from a road link or outgoing centroid 
    # connector onto successor links at a given time step is bounded by the sending flow for that 
    # destination on that link. The sending flow for a given destination on a link equals the upstream 
    # cumulative count for that destination evaluated at a time equal to the current time step minus 
    # the free-flow travel time of the link plus one, minus the downstream cumulative count for that 
    # destination at the current time step."
    #
    # sending_flow(a, s, t) = N_U(a, s, t - fftt(a) + 1) - N_D(a, s, t)
    # sum_b y(a->b, s, t) <= sending_flow(a, s, t)
    #
    # Wait, but N_D here means downstream count of link a. Let me re-check.
    # "the downstream cumulative count for that destination at the current time step" - this is N_D 
    # of the SAME link a.
    #
    # Hmm, but statement 2 says N_D(b) tracks inflow to b, not outflow from b.
    # 
    # Oh wait, I think I misread. Let me re-read statement 2:
    # "For each road link and each incoming centroid connector, the downstream cumulative count..."
    # This is about road links AND incoming CCs.
    # Statement 1 is about road links AND outgoing CCs.
    #
    # A road link appears in BOTH statements. So for a road link a:
    # N_U(a, s, t+1) = N_U(a, s, t) + sum_b y(a->b, s, t)  [outgoing from a]
    # N_D(a, s, t+1) = N_D(a, s, t) + sum_c y(c->a, s, t)  [incoming to a]
    #
    # OK so this makes N_U track exits and N_D track entries, which is the OPPOSITE of convention.
    # Or maybe it's: N_U is at the upstream end (entry), N_D is at the downstream end (exit).
    # 
    # Actually, I think the convention might be:
    # - Vehicles turn from predecessor links ONTO this link at the upstream end -> this is tracked by N_D
    #   (downstream count of the incoming flow process?)
    # - Vehicles turn from this link onto successor links at the downstream end -> tracked by N_U
    #
    # I'm confusing myself. Let me just follow the math exactly as stated.
    # 
    # N_U(a, s, t+1) = N_U(a, s, t) + [outgoing turning flow from a for dest s at t]
    # N_D(a, s, t+1) = N_D(a, s, t) + [incoming turning flow to a for dest s at t]
    #
    # Sending flow for link a: SF(a, s, t) = N_U(a, s, t - fftt + 1) - N_D(a, s, t)  (wait, this uses N_D of same link)
    # Hmm but that would be: vehicles that entered a by time t-fftt+1 (tracked in N_U? no N_U tracks exits)
    #
    # I think there might be a different reading. Let me try:
    # Maybe "from that link onto all of its successor links" means: the turning flow comes FROM 
    # predecessor links and goes ONTO "that link" (which is a successor). No, that doesn't parse.
    #
    # Let me try yet another reading of statement 1:
    # "the upstream cumulative count for [link L] at t+1 = current + turning flow from [link L] 
    #  onto all of its successor links"
    # So "from [link L]" and "onto successors of [link L]". The turning flow exits link L.
    # N_U(L) tracks cumulative exits from L... at the upstream end?
    #
    # I think the naming might be: N_U is the cumulative count at the Upstream boundary of the NEXT 
    # link (i.e., vehicles entering the successor), which equals vehicles exiting this link.
    # And N_D is the cumulative count at the Downstream boundary of the previous link, which equals 
    # vehicles entering this link.
    #
    # OK, I think I've been overthinking this. Let me just use the exact mathematical relationships 
    # as stated and not worry about physical interpretation.
    #
    # For the objective: "total vehicle-time spent on links, computed by summing over all links, 
    # all destinations, and all time steps the difference between the upstream and downstream 
    # cumulative counts"
    # obj_veh = sum_{link a, dest s, time t} (N_U(a,s,t) - N_D(a,s,t))
    #
    # Actually wait. N_U tracks exits, N_D tracks entries. N_U - N_D would be exits minus entries, 
    # which doesn't directly give vehicles on the link.
    # 
    # Hmm, or maybe I have the statements backwards. Let me try:
    # 
    # N_U(a, s, t+1) = N_U(a, s, t) + [flow entering a at its upstream end at time t]
    # N_D(a, s, t+1) = N_D(a, s, t) + [flow exiting a at its downstream end at time t]
    #
    # Then N_U - N_D = vehicles entered but not yet exited = vehicles ON the link.
    # The sending flow = N_U(a, s, t - fftt + 1) - N_D(a, s, t) makes sense: 
    #   vehicles that entered at least fftt-1 steps ago minus those that already exited.
    #
    # For this to work with the update equations:
    # "upstream count plus turning flow from that link onto successors" = entries into the link??
    # "from that link" means from link a... onto successors... that's exits from a.
    #
    # UNLESS "from that link" modifies the flows' direction as "vehicles that flow from [pred] and 
    # enter that link". But grammatically "from that link onto successors" means leaving the link.
    #
    # I wonder if there's an error in my reading. Let me try the other interpretation:
    # N_U(a, s, t+1) = N_U(a, s, t) + [incoming turning flow to a for dest s at t]
    # N_D(a, s, t+1) = N_D(a, s, t) + [outgoing turning flow from a for dest s at t]
    #
    # Then N_U - N_D = entries - exits = vehicles on link. Sending flow = N_U(a,s,t-fftt+1) - N_D(a,s,t)
    # = vehicles entered by t-fftt+1 minus exited by t. This makes sense.
    #
    # But then statement 1 should say "turning flow from predecessor links onto that link" for N_U,
    # and statement 2 should say "turning flow from that link onto successor links" for N_D.
    #
    # Let me re-read statement 1 one final time, very carefully:
    # "For each road link and each outgoing centroid connector, the upstream cumulative count for a 
    # given destination at the next time step equals the current upstream count plus the total 
    # turning flow of vehicles with that destination FROM THAT LINK onto all of ITS SUCCESSOR LINKS 
    # at the current time step."
    #
    # Hmm. So statement 1 applies to "road links AND outgoing centroid connectors".
    # An outgoing CC goes from centroid to junction. Its successor links are road links at the junction.
    # N_U of the outgoing CC increases by vehicles turning from the CC onto successor links.
    # Those vehicles are EXITING the outgoing CC. So N_U of the CC tracks exits.
    #
    # Statement 2: "For each road link and each incoming centroid connector, the downstream 
    # cumulative count for a given destination at the next time step equals the current downstream 
    # count plus the total turning flow of vehicles with that destination FROM ALL PREDECESSOR LINKS 
    # ONTO THAT LINK at the current time step."
    # An incoming CC goes from junction to centroid. Its predecessor links are road links at the junction.
    # N_D of the incoming CC increases by vehicles turning from predecessors onto the CC.
    # Those vehicles are ENTERING the incoming CC. So N_D of the CC tracks entries.
    #
    # For a road link a (from junction u to junction v):
    # Statement 1: N_U(a, s, t+1) = N_U(a, s, t) + sum_b y(a->b, s, t) [exits from a at downstream end]
    # Statement 2: N_D(a, s, t+1) = N_D(a, s, t) + sum_c y(c->a, s, t) [entries to a at upstream end]
    #
    # So N_U tracks downstream exits, N_D tracks upstream entries??
    # Then N_U - N_D = exits - entries (negative of vehicles on link). 
    # The objective sum(N_U - N_D) would be negative... that doesn't make sense for minimization.
    #
    # Unless in the objective, it's N_D - N_U? Let me re-read:
    # "the total vehicle-time spent on links, computed by summing over all links, all destinations, 
    # and all time steps the difference between the upstream and downstream cumulative counts"
    # "difference between upstream and downstream" = N_U - N_D
    #
    # Hmm. If N_U = exits, N_D = entries, then N_U <= N_D (can't exit more than entered). 
    # N_U - N_D <= 0. Minimizing a negative sum makes no sense.
    #
    # So I think the conventional interpretation must apply:
    # N_U = upstream cumulative count = entries at upstream end
    # N_D = downstream cumulative count = exits at downstream end
    # N_U - N_D >= 0 = vehicles currently on the link
    #
    # And the update equations must be:
    # Statement 1 about N_U: relates to vehicles entering the link from predecessors
    # Statement 2 about N_D: relates to vehicles exiting the link to successors
    #
    # I think the problem statement's wording is awkward but means:
    # Statement 1: For link L, N_U(L, s, t+1) = N_U(L, s, t) + [turning flow from predecessors onto L for dest s at t]
    # The phrase "from that link" might actually refer to the turning flow mechanism, not the direction.
    #
    # Actually, let me try yet another reading:
    # Statement 1: "the upstream cumulative count [of link L]... equals current count plus the total 
    # turning flow... from that link onto all of its successor links"
    # 
    # What if "that link" refers not to L but to the predecessor? Like "turning flow from [predecessor] 
    # [that is a link] onto [L which is one of] its successor links"?
    # That's a stretch grammatically but would make physical sense.
    #
    # Actually, I just realized there might be a simpler resolution. Let me re-read:
    # Statement 1 is about "road links AND outgoing centroid connectors"
    # Statement 2 is about "road links AND incoming centroid connectors"
    #
    # For an outgoing CC (centroid -> junction):
    # - Its upstream end is at the centroid
    # - Vehicles enter it from the centroid
    # - Its downstream end is at a junction
    # - Vehicles exit it to road links at the junction (successor links)
    #
    # Statement 1 says N_U of outgoing CC increases by turning flow from CC to successors = EXIT flow.
    # But N_U should be at the upstream (centroid) end...
    #
    # Oh wait. Maybe in LTM, N_U and N_D don't refer to ends of the link but to the upstream and 
    # downstream counts in the Newell-Daganzo sense?
    # In the triangular fundamental diagram model:
    # N_U(t) = cumulative number of vehicles that have "entered" (passed the upstream point)
    # N_D(t) = cumulative number that have "exited" (passed the downstream point)
    #
    # And the update is:
    # N_U(L, t+1) = N_U(L, t) + inflow to L at time t (entering upstream end)
    # N_D(L, t+1) = N_D(L, t) + outflow from L at time t (exiting downstream end)
    #
    # Turning flows from predecessors to L represent inflow to L = update to N_U(L)
    # Turning flows from L to successors represent outflow from L = update to N_D(L)
    #
    # But the problem says the OPPOSITE mapping (statement 1 maps outflow to N_U, statement 2 maps 
    # inflow to N_D).
    #
    # I think the problem statement might have swapped the names, OR I'm misreading.
    #
    # Let me try one more time with the EXACT text:
    #
    # "For each road link and each outgoing centroid connector, the upstream cumulative count for a 
    # given destination at the next time step equals the current upstream count plus the total turning 
    # flow of vehicles with that destination from that link onto all of its successor links at the 
    # current time step."
    #
    # I'll parse this as: For link L (road or outgoing CC):
    # N_U(L, s, t+1) = N_U(L, s, t) + sum_{B in succ(L)} y(L, B, s, t)
    # where y(L, B, s, t) is the turning flow from L to B.
    # This flow exits L's downstream end.
    # So N_U tracks cumulative EXITS from L. (Despite being called "upstream".)
    #
    # "For each road link and each incoming centroid connector, the downstream cumulative count for a 
    # given destination at the next time step equals the current downstream count plus the total 
    # turning flow of vehicles with that destination from all predecessor links onto that link at the 
    # current time step."
    #
    # For link L (road or incoming CC):
    # N_D(L, s, t+1) = N_D(L, s, t) + sum_{A in pred(L)} y(A, L, s, t)
    # This flow enters L's upstream end.
    # So N_D tracks cumulative ENTRIES to L. (Despite being called "downstream".)
    #
    # Then: N_D - N_U = entries - exits = vehicles on link >= 0.
    # Objective: sum(N_U - N_D) = sum(exits - entries) < 0. That's weird.
    #
    # UNLESS the objective says "upstream minus downstream" meaning N_U - N_D, and since this is 
    # negative, minimizing it would mean maximizing throughput. But that doesn't match "minimize 
    # total system travel time."
    #
    # I think the issue is that cumulative counts in LTM are not "entries" and "exits" but rather 
    # Newell cumulative counts at the upstream and downstream boundaries. In LTM:
    # - N(upstream, t) increases when a vehicle passes the upstream boundary (enters the link)
    # - N(downstream, t) increases when a vehicle passes the downstream boundary (exits the link)
    #
    # And the number of vehicles on the link = N(upstream, t) - N(downstream, t).
    # Total vehicle-time = sum over t of [N(upstream, t) - N(downstream, t)].
    #
    # So N_U should track entries and N_D should track exits.
    #
    # Given the problem's word choice, I think "from that link onto successors" in statement 1 
    # actually means: the turning flow that enters the UPSTREAM end of that link FROM predecessors.
    # Wait no, "onto successors" clearly means going TO successors.
    #
    # OK HERE IS MY NEW THEORY: the problem defines N_U on the link that the flow is COMING FROM 
    # and N_D on the link the flow is GOING TO.
    #
    # N_U of link A = cumulative flow that has exited A at A's downstream end
    # N_D of link A = cumulative flow that has entered A at A's upstream end
    # 
    # And the sending flow = N_U(A, s, t - fftt + 1) - N_D(A, s, t)
    #                       = [entered A by t-fftt+1] - [exited A by t] ??? No, that doesn't work.
    #
    # Let me try: N_D tracks entries, N_U tracks exits.
    # Sending flow = N_D(A, s, t - fftt + 1) - N_U(A, s, t)
    #             = [entered A by t-fftt+1] - [exited A by t]
    # But the problem says it's N_U(A, s, t-fftt+1) - N_D(A, s, t).
    # With my mapping: = [exited A by t-fftt+1] - [entered A by t]. Negative. No good.
    #
    # Final attempt: standard LTM interpretation:
    # N_U(A, s, t) = cumulative entries to A for dest s by time t
    # N_D(A, s, t) = cumulative exits from A for dest s by time t
    #
    # Entries to A = turning flows from predecessors to A = sum_{P} y(P->A, s, t)
    # Exits from A = turning flows from A to successors = sum_{B} y(A->B, s, t)
    #
    # N_U(A, s, t+1) = N_U(A, s, t) + sum_{P} y(P->A, s, t)  [entries]
    # N_D(A, s, t+1) = N_D(A, s, t) + sum_{B} y(A->B, s, t)  [exits]
    #
    # Sending flow = N_U(A, s, t - fftt + 1) - N_D(A, s, t)
    #             = [entered by t-fftt+1] - [exited by t]
    # This is the number of vehicles that entered A early enough to have reached the downstream end 
    # by time t (given free-flow travel time fftt), minus those that already exited. Makes perfect sense!
    #
    # Now matching to problem statement:
    # Statement 1 about N_U says: N_U = current + "turning flow from that link onto successors"
    #   = current + exits from A. But N_U should track entries!
    #   Contradiction.
    # Statement 2 about N_D says: N_D = current + "turning flow from predecessors onto that link"
    #   = current + entries to A. But N_D should track exits!
    #   Contradiction.
    #
    # SO: Either the problem statement has swapped the names, or my understanding of the standard 
    # LTM is wrong, or I need to read the problem statement differently.
    #
    # Given that the objective says "sum of N_U - N_D" for vehicle time:
    # If N_U = entries, N_D = exits, then N_U - N_D = vehicles on link >= 0. Correct for vehicle time.
    # If N_U = exits, N_D = entries, then N_U - N_D <= 0. Incorrect for vehicle time.
    #
    # For the objective to make sense, N_U must track entries and N_D must track exits.
    #
    # Therefore, I will use the standard interpretation:
    # N_U(A, s, t+1) = N_U(A, s, t) + [inflow to A for dest s at t] (from predecessors)
    # N_D(A, s, t+1) = N_D(A, s, t) + [outflow from A for dest s at t] (to successors)
    #
    # And interpret the problem statements accordingly (perhaps the wording is just confusing).

    # Let me also re-read the centroid connector parts:
    # "For incoming centroid connectors, the downstream cumulative count for each destination at the 
    # next time step simply equals the upstream cumulative count at the current time step."
    # N_D(incoming_CC, s, t+1) = N_U(incoming_CC, s, t)
    # With standard interpretation: exits by t+1 = entries by t. Meaning there's a 1-step delay, 
    # which is the free-flow travel time of 1 step. Makes sense!
    #
    # "Vehicles cannot pass through an incoming centroid connector unless their destination matches 
    # the centroid at which that connector terminates"
    # Only destination = terminal centroid allowed on incoming CCs. Already handled above.

    # "The upstream cumulative count on an outgoing centroid connector for a given destination at the 
    # next time step equals the current count plus the number of vehicles departing the centroid along 
    # that connector toward that destination."
    # N_U(outgoing_CC(c,j), s, t+1) = N_U(outgoing_CC(c,j), s, t) + yc(c, j, s, t)
    # Entries to outgoing CC = yc. Makes sense.

    # OK so with standard interpretation, let me re-derive all constraints:
    #
    # For ROAD LINKS (from junction u to junction v):
    # N_U(u,v,s,t+1) = N_U(u,v,s,t) + sum_{pred links P ending at u} y(P.from, u, v, s, t)
    # N_D(u,v,s,t+1) = N_D(u,v,s,t) + sum_{succ links B starting at v} y(u, v, B.to, s, t)
    #
    # For OUTGOING CCs (centroid c to junction j):
    # N_U(c,j,s,t+1) = N_U(c,j,s,t) + yc(c,j,s,t)  [vehicles depart centroid onto CC]
    # N_D(c,j,s,t+1) = N_D(c,j,s,t) + sum_{succ links B at j} y(c,j,B.to,s,t)
    #   [vehicles exit CC to road links - these are the turning flows from the CC]
    #
    # For INCOMING CCs (junction j to centroid c):
    # N_U(j,c,s,t+1) = N_U(j,c,s,t) + sum_{pred links P at j} y(P.from, j, c, s, t)
    #   [vehicles enter CC from road links]
    # N_D(j,c,s,t+1) = N_U(j,c,s,t)  [special rule for incoming CCs]
    #   (only for s = c)
    #
    # Sending flow for link A with fftt:
    # SF(A, s, t) = N_U(A, s, t - fftt + 1) - N_D(A, s, t)
    # sum_{succ B} y(A->B, s, t) <= SF(A, s, t)
    # (for t >= fftt, since turning flows are 0 before fftt)
    #
    # Capacity constraints on road links:
    # sum_{s} sum_{succ B} y(A->B, s, t) <= capacity  [outgoing flow <= capacity]
    # sum_{s} sum_{pred P} y(P->A, s, t) <= capacity  [incoming flow <= capacity]
    #
    # Congested wave constraint on road links:
    # Let cwtt = length / congested_wave_speed (in time steps)
    # sum_{s} sum_{pred P} y(P->A, s, t) <= sum_s [N_D(A, s, t - cwtt + 1) - N_U(A, s, t)] + jam_density * length
    # Wait, let me re-read:
    # "the total incoming flow is limited by a congested-wave bound: the sum of all incoming turning 
    # flows may not exceed the sum over all destinations of the difference between the upstream 
    # cumulative count evaluated at a time equal to the current time step minus the ratio of link 
    # length to congested wave speed plus one and the downstream cumulative count at the current time 
    # step, plus the maximum vehicle occupancy of that link."
    #
    # Hmm: "difference between the upstream cumulative count evaluated at [t - cwtt + 1] and the 
    # downstream cumulative count at [t], plus the maximum vehicle occupancy"
    # 
    # With standard interpretation:
    # sum_s inflow(A, s, t) <= sum_s [N_U(A, s, t - cwtt + 1) - N_D(A, s, t)] + N_jam
    # Hmm, but N_U(t-cwtt+1) - N_D(t) isn't "the difference between upstream evaluated at ... 
    # and downstream at ..." in the problem's naming.
    #
    # The problem says "difference between the upstream cumulative count [at t-cwtt+1] and the 
    # downstream cumulative count [at t]". Using the problem's N_U and N_D labels:
    # = N_U(A, s, t-cwtt+1) - N_D(A, s, t)
    #
    # But we established that in the problem's naming, N_U tracks... well, let me just define 
    # my own variable names to avoid confusion and map to the output format at the end.
    #
    # Let me define:
    # CU(A, s, t) = cumulative inflow (entries) to link A for dest s by time t [= problem's N_U if standard]
    # CD(A, s, t) = cumulative outflow (exits) from link A for dest s by time t [= problem's N_D if standard]
    #
    # Then:
    # Vehicles on link A at time t = sum_s [CU(A,s,t) - CD(A,s,t)]
    # Objective vehicle time = sum_{A,s,t} [CU(A,s,t) - CD(A,s,t)]
    # This should be non-negative and we minimize it.
    #
    # Updates for road links:
    # CU(A, s, t+1) = CU(A, s, t) + inflow(A, s, t)
    # CD(A, s, t+1) = CD(A, s, t) + outflow(A, s, t)
    #
    # Sending flow (outflow constraint):
    # outflow(A, s, t) <= CU(A, s, t - fftt + 1) - CD(A, s, t)
    # [vehicles entered early enough minus already exited]
    #
    # Receiving flow / congested wave constraint:
    # sum_s inflow(A, s, t) <= sum_s [CD(A, s, t - cwtt + 1) - CU(A, s, t)] + N_jam
    # Wait, let me re-derive. In LTM, the receiving flow is:
    # R(t) = N_jam - [N_U(t) - N_D(t - L/w + 1)] where w is congested wave speed
    # With my notation CU = entries, CD = exits:
    # R(t) = N_jam - [CU(A, t) - CD(A, t - cwtt + 1)]
    #       = N_jam - CU(A, t) + CD(A, t - cwtt + 1)
    #       = [CD(A, t - cwtt + 1) - CU(A, t)] + N_jam
    # sum_s inflow(A, s, t) <= sum_s [CD(A, s, t - cwtt + 1) - CU(A, s, t)] + N_jam
    #
    # Hmm, but the problem says:
    # "sum over all destinations of the difference between the upstream cumulative count evaluated 
    # at t - cwtt + 1 and the downstream cumulative count at t, plus N_jam"
    # = sum_s [N_U(A, s, t-cwtt+1) - N_D(A, s, t)] + N_jam
    #
    # If in the problem's naming N_U = my CD (exits) and N_D = my CU (entries), then:
    # = sum_s [CD(A, s, t-cwtt+1) - CU(A, s, t)] + N_jam
    # Which matches the standard LTM receiving flow!
    #
    # So the problem's naming convention is:
    # problem's N_U = cumulative EXITS (downstream outflow) = my CD
    # problem's N_D = cumulative ENTRIES (upstream inflow) = my CU
    # Despite the U/D naming, the updates make sense with this.
    #
    # Let me verify with the sending flow:
    # Problem: SF = problem's N_U(A, s, t-fftt+1) - problem's N_D(A, s, t)
    #        = CD(A, s, t-fftt+1) - CU(A, s, t)
    # But standard sending flow = CU(A, s, t-fftt+1) - CD(A, s, t)
    # These don't match!
    #
    # Hmm. Let me try yet another interpretation.
    # What if problem's N_U = my CU (entries) and problem's N_D = my CD (exits)?
    # Then the update equations from the problem are:
    # CU(A, s, t+1) = CU(A, s, t) + [turning flow from A to successors] = entries + exits?? No.
    #
    # I'm going in circles. Let me just go with the LITERAL problem statement and see if the math 
    # is self-consistent:
    #
    # DEFINE (following problem statement literally):
    # N_U(A, s, t+1) = N_U(A, s, t) + sum_B y(A->B, s, t)  [for road links, outgoing CCs]
    # N_D(A, s, t+1) = N_D(A, s, t) + sum_P y(P->A, s, t)  [for road links, incoming CCs]
    # (A is the link in question, P are predecessors, B are successors)
    #
    # Sending flow: outflow from A <= N_U(A, s, t-fftt+1) - N_D(A, s, t)
    # sum_B y(A->B, s, t) <= N_U(A, s, t-fftt+1) - N_D(A, s, t)
    # 
    # OK wait. The LHS is sum_B y(A->B, s, t) which also appears in the N_U update.
    # So N_U increases by exactly the outflow from A.
    # And N_D increases by exactly the inflow to A.
    # 
    # Let me check: is N_U(A, s, t-fftt+1) - N_D(A, s, t) non-negative?
    # At time 0: N_U = N_D = 0.
    # N_U increases by outflow, N_D increases by inflow.
    # For a link with fftt = 2:
    # t=0: inflow can occur (N_D increases), but outflow cannot (turning flows 0 for t < fftt).
    # So N_D can grow while N_U stays at 0 for t < fftt.
    # At t = fftt = 2: sending flow = N_U(A, s, 2-2+1) - N_D(A, s, 2) = N_U(A, s, 1) - N_D(A, s, 2)
    # N_U(A, s, 1) = 0 (no outflow at t=0 since t < fftt)
    # N_D(A, s, 2) could be positive (inflow at t=0 and t=1)
    # So sending flow = 0 - positive = negative! That's a problem - can't have negative bound on flow.
    #
    # This means with the literal reading, the sending flow could be negative, which is nonsensical.
    # So the literal reading must be wrong.
    #
    # With the swapped interpretation (N_U = entries, N_D = exits):
    # N_U(A, s, t+1) = N_U(A, s, t) + inflow to A
    # N_D(A, s, t+1) = N_D(A, s, t) + outflow from A
    # Sending flow: outflow from A <= N_U(A, s, t-fftt+1) - N_D(A, s, t)
    #  = entries by t-fftt+1 - exits by t
    # For t=fftt: = entries by 1 - exits by fftt
    # exits by fftt = 0 (can't exit before fftt), so = entries by 1 >= 0. Good!
    #
    # So the correct interpretation must be:
    # N_U(A, s, t+1) = N_U(A, s, t) + sum_P y(P->A, s, t)  [INFLOW = entries]
    # N_D(A, s, t+1) = N_D(A, s, t) + sum_B y(A->B, s, t)  [OUTFLOW = exits]
    #
    # And the problem statements 1 and 2 must be interpreted as:
    # Statement 1 (N_U): "turning flow from [predecessor links] onto [that link]" = INFLOW
    #   Reading: "turning flow... from [pred] link onto... [that link which is a successor of pred]"
    #   The "its" in "its successor links" refers to the predecessor, not to "that link".
    # Statement 2 (N_D): "turning flow from [that link] onto [successor links]" via predecessors
    #   Wait no, statement 2 says "from all predecessor links onto that link"
    #
    # Hmm, statement 2 literally says "from predecessors onto that link" which sounds like INFLOW to 
    # that link, which would go to N_D. But N_D should track exits...
    #
    # OK I think the problem statement may have swapped statements 1 and 2 or there's genuine 
    # confusion in the naming. Let me just go with what makes the math work:
    #
    # Define for my implementation:
    # N_up[A, s, t] = cumulative entries to link A  (problem output field: N_U)
    # N_dn[A, s, t] = cumulative exits from link A  (problem output field: N_D)
    #
    # Update:
    # N_up[A, s, t+1] = N_up[A, s, t] + inflow_to_A(s, t)
    # N_dn[A, s, t+1] = N_dn[A, s, t] + outflow_from_A(s, t)
    #
    # Sending: outflow_from_A(s, t) <= N_up[A, s, t - fftt + 1] - N_dn[A, s, t]
    # Receiving: sum_s inflow_to_A(s, t) <= sum_s [N_dn[A, s, t - cwtt + 1] - N_up[A, s, t]] + N_jam
    #
    # Objective: sum_{A,s,t} [N_up[A,s,t] - N_dn[A,s,t]] + waiting time
    #
    # For incoming CCs: N_dn[A, s, t+1] = N_up[A, s, t] (delayed by 1 step)
    # For outgoing CCs: N_up[A, s, t+1] = N_up[A, s, t] + yc(c, j, s, t)
    #
    # Parked vehicles:
    # p[c, t+1] = p[c, t] + arrivals - departures
    # arrivals at c at time t = sum of vehicles that exit incoming CCs to c at time t
    #   = difference in N_up of incoming CC for dest c between t+1 and t (flow entering incoming CC at t)
    # Wait, let me re-read: "the number of vehicles arriving at that centroid via incoming centroid 
    # connectors (measured as the difference between upstream and downstream cumulative counts for 
    # vehicles destined for that centroid)"
    # 
    # Hmm, "difference between upstream and downstream cumulative counts" - this is a stock, not a flow.
    # But p update is p[c,t+1] = p[c,t] + arrivals - departures.
    # If arrivals = N_up(incoming_CC, c, t) - N_dn(incoming_CC, c, t) [with my naming], that's the 
    # vehicles currently on the incoming CC destined for c. But that's a stock, and adding it to p 
    # each step would double-count.
    #
    # Actually wait: let me re-read "the number of vehicles parked at a centroid at the next time 
    # step equals the current parked count plus the number of vehicles arriving at that centroid via 
    # incoming centroid connectors (measured as the difference between upstream and downstream 
    # cumulative counts for vehicles destined for that centroid) minus the total number of vehicles 
    # departing that centroid via outgoing centroid connectors toward all destinations."
    #
    # I think "arriving" means the FLOW of vehicles completing traversal of the incoming CC.
    # "measured as the difference between upstream and downstream cumulative counts" might mean 
    # the CHANGE, i.e., the flow that crossed.
    #
    # Hmm, but "difference between upstream and downstream cumulative counts" at a given time is a 
    # stock (vehicles on link), not a flow. Unless it means the difference in the CHANGES.
    #
    # Actually, for an incoming CC with fftt=1 and the special rule N_dn(t+1) = N_up(t):
    # Flow exiting the CC at time t (arriving at centroid) = N_dn(t+1) - N_dn(t) = N_up(t) - N_dn(t)
    # So the flow that arrives at centroid between t and t+1 equals N_up(t) - N_dn(t)!
    # The "difference between upstream and downstream cumulative counts" IS the arrival flow.
    # This is because of the special 1-step delay property.
    #
    # So: arrivals_c(t) = sum_{incoming CC to c} [N_up(CC, c, t) - N_dn(CC, c, t)]
    # departures_c(t) = sum_{outgoing CC from c} sum_s yc(c, j, s, t)
    #
    # p[c, t+1] = p[c, t] + arrivals_c(t) - departures_c(t)
    # 
    # With my naming where N_up = entries and N_dn = exits:
    # arrivals_c(t) = sum_{incoming CC L to c} [N_up(L, c, t) - N_dn(L, c, t)]
    # = vehicles on the incoming CC at time t (which will arrive at t+1 due to 1-step travel time)
    #
    # But wait, the equation is p(c, t+1) = p(c, t) + arrivals(t) - departures(t)
    # and arrivals(t) = N_up(L, c, t) - N_dn(L, c, t).
    # N_up(L, c, t) - N_dn(L, c, t) = entries by t - exits by t = vehicles currently on link.
    # For a 1-step link with N_dn(t+1) = N_up(t): N_dn(t) = N_up(t-1).
    # N_up(t) - N_dn(t) = N_up(t) - N_up(t-1) = inflow at t-1. 
    # Wait, N_up(t) = N_up(t-1) + inflow(t-1).
    # So N_up(t) - N_up(t-1) = inflow(t-1).
    # And the arrival flow from this CC at time t = inflow(t-1) = vehicles that entered the CC at t-1,
    # which arrive at the centroid at t. That's the flow that departs the CC at its downstream end 
    # at time t = N_dn(t) - N_dn(t-1) = N_up(t-1) - N_up(t-2) = inflow(t-2)... hmm, that's different.
    #
    # Actually N_dn(t+1) = N_up(t), so:
    # Exit flow at time t = N_dn(t+1) - N_dn(t) = N_up(t) - N_up(t-1) = inflow(t-1).
    # Hmm wait, the exit flow between t and t+1 is N_dn(t+1) - N_dn(t). That exit happens at time 
    # t+1 essentially. But vehicles that entered at t-1 (inflow at t-1) would arrive after fftt=1 
    # steps, so at time t. Exit at time t = N_dn(t) - N_dn(t-1) = N_up(t-1) - N_up(t-2) = inflow(t-2).
    # That means a vehicle entering at t-2 exits at t, taking 2 steps. But fftt = 1.
    #
    # I think there's an off-by-one issue. Let me be more careful.
    # N_dn(CC, c, t+1) = N_up(CC, c, t)
    # So N_dn(CC, c, 1) = N_up(CC, c, 0) = 0
    # N_dn(CC, c, 2) = N_up(CC, c, 1) = N_up(CC, c, 0) + inflow(0) = inflow(0)
    # So the first vehicles to enter the CC at time 0 are counted as exited at time 2.
    # But with fftt=1, they should exit at time 1.
    #
    # Hmm, let me think about what "flow at time t" means in the discrete LTM. I think the 
    # cumulative count N(t) means cumulative count BY the beginning of step t (or at time t).
    # The flow during step t (from t to t+1) is N(t+1) - N(t).
    # If fftt = 1, a vehicle entering at step t should exit at step t+1.
    # Entering at step t means it contributes to the count between t and t+1: N_up(t+1) - N_up(t).
    # Exiting at step t+1 means: N_dn(t+2) - N_dn(t+1).
    # With N_dn(t+1) = N_up(t): N_dn(t+2) - N_dn(t+1) = N_up(t+1) - N_up(t) = inflow at t.
    # So exit at step t+1 = inflow at step t. Travel time = 1 step. Correct!
    #
    # Now for arrivals in the parked vehicle equation:
    # p(c, t+1) = p(c, t) + arrivals(t) - departures(t)
    # arrivals(t) should be the flow of vehicles arriving at centroid c during step t.
    # = exit flow from incoming CC during step t
    # = N_dn(CC, c, t+1) - N_dn(CC, c, t)
    #
    # The problem says arrivals(t) = N_up(CC, c, t) - N_dn(CC, c, t)
    # Let's check: N_up(CC, c, t) - N_dn(CC, c, t).
    # Using N_dn(t) = N_up(t-1) (from the special rule shifted by 1):
    # = N_up(CC, c, t) - N_up(CC, c, t-1) = inflow during step t-1.
    # And exit flow during step t = N_dn(t+1) - N_dn(t) = N_up(t) - N_up(t-1) = inflow at t-1.
    # So indeed, N_up(t) - N_dn(t) = exit flow during step t = arrivals at step t. 
    #
    # But note this is also = vehicles on the link at time t (entered but not exited).
    # With N_dn(t) = N_up(t-1): on link = N_up(t) - N_up(t-1) = last step's inflow.
    # And since fftt=1, those vehicles will all exit in this step, so they're transiting.
    # Makes sense!
    #
    # Great. Now let me also check the problem's statement about N_U and N_D more carefully by 
    # looking at the incoming CC:
    # Problem says: "For incoming centroid connectors, the downstream cumulative count for each 
    # destination at the next time step simply equals the upstream cumulative count at the current 
    # time step."
    # N_D(CC, s, t+1) = N_U(CC, s, t)
    #
    # In my notation: if I set N_D of the problem = my N_dn:
    # N_dn(CC, c, t+1) = N_U_problem(CC, c, t)
    # For the special rule to work (N_dn(t+1) = N_up(t)), we need:
    # N_U_problem = my N_up (entries).
    # And then N_D_problem = my N_dn (exits).
    # 
    # So the problem's N_U = entries and N_D = exits. STANDARD INTERPRETATION.
    # And the update equations in the problem must be:
    # N_U (entries): increases with inflow (from predecessors)
    # N_D (exits): increases with outflow (to successors)
    #
    # Then statement 1 must mean inflow, and statement 2 must mean outflow, even though the 
    # wording seems to say the opposite. I suspect the problem statement has confusing wording 
    # where "from that link onto its successor links" in statement 1 actually describes the 
    # turning motion at the upstream node of the link, where vehicles come from a predecessor 
    # link and turn onto the current link (which is a "successor" from the predecessor's point of view).
    #
    # Regardless, I'll use:
    # N_U = entries (inflow at upstream end) = cumulative vehicles entering the link
    # N_D = exits (outflow at downstream end) = cumulative vehicles exiting the link

    # Now let me implement this properly.
    # 
    # For turning flows y(from_node, via_node, to_node, dest, time):
    # This represents vehicles turning from link (from_node -> via_node) to link (via_node -> to_node).
    # = outflow from link (from_node -> via_node) = inflow to link (via_node -> to_node)
    # The turning happens at node via_node (a junction).

    # Let me rebuild the constraint formulation properly.
    
    # For each link A = (u, v):
    # inflow(A, s, t) = sum of y from predecessors to A for dest s at t
    #                  = sum_{P ending at u} y(P.from, u, v, s, t)  [for road links]
    #                  = yc(c, v, s, t) if A is an outgoing CC (c, v) [special: from centroid]
    # 
    # Wait, for outgoing CC (c, j): the inflow is yc(c, j, s, t).
    # There's no turning flow from predecessor links since the upstream end is a centroid.
    # 
    # outflow(A, s, t) = sum of y from A to successors for dest s at t
    #                   = sum_{B starting at v} y(u, v, B.to, s, t)  [for road links]
    #
    # For incoming CC (j, c): the outflow exits at centroid c. There are no successor links 
    # from centroid c (except outgoing CCs, but vehicles don't turn from incoming CC directly to 
    # outgoing CC - they park first). So outflow from incoming CC is handled by the special rule.
    #
    # Actually, for incoming CC, the downstream end is a centroid. Vehicles exiting the CC arrive 
    # at the centroid and park. The outflow is captured by N_D update.
    # N_D(incoming_CC, c, t+1) = N_U(incoming_CC, c, t) [special rule]
    # So we don't need explicit outflow turning variables for incoming CCs.

    # Let me now carefully set up all the variables and constraints.
    # I'll use the y_vars already created (turning at junctions) and yc_vars (centroid departures).
    
    # For the N_U and N_D variables, I'll use NU_vars and ND_vars as created.
    # NU = entries (inflow), ND = exits (outflow).

    # But wait - I mapped dests_for_link assuming both endpoints need to reach s.
    # For outgoing CCs and incoming CCs, the dests are limited.
    # For incoming CCs to centroid c: only dest c. (Already handled.)
    # For outgoing CCs from centroid c: any destination reachable from the junction end.

    # Let me now reformulate properly. I need to carefully separate the link types.
    
    # Helper: for a given link, get its relevant destinations
    def get_link_dests(l):
        u, v = l['from'], l['to']
        if l['type'] == 'centroid_connector' and v in centroid_set:
            # Incoming CC to centroid v: only dest v
            return [v] if v in dest_set else []
        else:
            return dests_for_link.get((u, v), [])

    # Rebuild y_vars, NU_vars, ND_vars more carefully
    m = gp.Model("SAV")
    m.Params.TimeLimit = max(1, time_limit - 30)
    m.Params.Threads = 1
    m.Params.Method = 2  # Barrier
    m.Params.OutputFlag = 1

    # Precompute turn info
    # At each junction v, turns (il, ol) with common destinations
    turn_dests = {}  # (il_from, v, ol_to) -> list of dests
    all_turns = []
    for v in junctions:
        in_links_v = incoming[v]
        out_links_v = outgoing[v]
        for il in in_links_v:
            for ol in out_links_v:
                u = il['from']
                w = ol['to']
                d_il = set(get_link_dests(il))
                d_ol = set(get_link_dests(ol))
                common = sorted(d_il & d_ol)
                if common:
                    turn_dests[(u, v, w)] = common
                    all_turns.append((u, v, w, il, ol))

    # y variables
    y_vars = {}
    for (u, v, w, il, ol) in all_turns:
        fftt_il = il['free_flow_travel_time_steps']
        for s in turn_dests[(u, v, w)]:
            for t in range(fftt_il, T):
                y_vars[(u, v, w, s, t)] = m.addVar(lb=0)

    # yc variables
    yc_vars = {}
    for c in centroids:
        for l in outgoing_cc_by_centroid[c]:
            j = l['to']
            dests_l = get_link_dests(l)
            for s in dests_l:
                for t in range(T):
                    yc_vars[(c, j, s, t)] = m.addVar(lb=0)

    # N_U, N_D variables (only where needed)
    NU_vars = {}
    ND_vars = {}
    for l in links_list:
        u, v = l['from'], l['to']
        dests_l = get_link_dests(l)
        for s in dests_l:
            for t in range(T + 1):
                NU_vars[(u, v, s, t)] = m.addVar(lb=0)
                ND_vars[(u, v, s, t)] = m.addVar(lb=0)

    # p, e, omega variables
    p_vars = {}
    for c in centroids:
        for t in range(T + 1):
            p_vars[(c, t)] = m.addVar(lb=0)

    e_vars = {}
    for (o, d) in od_pairs:
        for t in range(T):
            e_vars[(o, d, t)] = m.addVar(lb=0)

    omega_vars = {}
    for (o, d) in od_pairs:
        for t in range(T + 1):
            omega_vars[(o, d, t)] = m.addVar(lb=0)

    m.update()
    
    elapsed = time.time() - start_time
    print(f"Variables created: {m.NumVars}, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # CONSTRAINTS
    
    # 1. Initial conditions
    for l in links_list:
        u, v = l['from'], l['to']
        for s in get_link_dests(l):
            m.addConstr(NU_vars[(u, v, s, 0)] == 0)
            m.addConstr(ND_vars[(u, v, s, 0)] == 0)

    # 2. N_U (entries) update for all links
    for l in links_list:
        u, v = l['from'], l['to']
        dests_l = get_link_dests(l)
        
        if l['type'] == 'centroid_connector' and u in centroid_set:
            # Outgoing CC: entries come from centroid departure
            c = u
            j = v
            for s in dests_l:
                for t in range(T):
                    yc_var = yc_vars.get((c, j, s, t), None)
                    if yc_var is not None:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)] + yc_var)
                    else:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])
        elif l['type'] == 'centroid_connector' and v in centroid_set:
            # Incoming CC: entries come from turning flows at junction u
            j = u
            c = v
            for s in dests_l:  # only s = c
                # Inflow = sum of y(pred_of_j, j, c, s, t) for all pred links ending at j
                for t in range(T):
                    inflow_terms = []
                    for il in incoming[j]:
                        key = (il['from'], j, c, s, t)
                        if key in y_vars:
                            inflow_terms.append(y_vars[key])
                    if inflow_terms:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)] + gp.quicksum(inflow_terms))
                    else:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])
        else:
            # Road link: entries come from turning flows from predecessors
            for s in dests_l:
                for t in range(T):
                    inflow_terms = []
                    for il in incoming[u]:
                        key = (il['from'], u, v, s, t)
                        if key in y_vars:
                            inflow_terms.append(y_vars[key])
                    if inflow_terms:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)] + gp.quicksum(inflow_terms))
                    else:
                        m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])

    elapsed = time.time() - start_time
    print(f"N_U constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 3. N_D (exits) update
    for l in links_list:
        u, v = l['from'], l['to']
        dests_l = get_link_dests(l)
        
        if l['type'] == 'centroid_connector' and v in centroid_set:
            # Incoming CC: special rule N_D(t+1) = N_U(t)
            for s in dests_l:
                for t in range(T):
                    m.addConstr(ND_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])
        elif l['type'] == 'centroid_connector' and u in centroid_set:
            # Outgoing CC: exits are turning flows from CC to road links at junction v
            c = u
            j = v
            fftt_l = l['free_flow_travel_time_steps']
            for s in dests_l:
                for t in range(T):
                    outflow_terms = []
                    for ol in outgoing[j]:
                        key = (c, j, ol['to'], s, t)
                        if key in y_vars:
                            outflow_terms.append(y_vars[key])
                    if outflow_terms:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)] + gp.quicksum(outflow_terms))
                    else:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)])
        else:
            # Road link: exits are turning flows from this link to successors
            for s in dests_l:
                for t in range(T):
                    outflow_terms = []
                    for ol in outgoing[v]:
                        key = (u, v, ol['to'], s, t)
                        if key in y_vars:
                            outflow_terms.append(y_vars[key])
                    if outflow_terms:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)] + gp.quicksum(outflow_terms))
                    else:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)])

    elapsed = time.time() - start_time
    print(f"N_D constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 4. Sending flow constraints
    # For road links and outgoing CCs:
    # sum_B y(u,v,B.to,s,t) <= N_U(u,v,s,t-fftt+1) - N_D(u,v,s,t)
    for l in links_list:
        u, v = l['from'], l['to']
        if l['type'] == 'centroid_connector' and v in centroid_set:
            continue  # incoming CC: no outgoing turning flows needed
        
        fftt = l['free_flow_travel_time_steps']
        dests_l = get_link_dests(l)
        
        for s in dests_l:
            for t in range(fftt, T):
                outflow_terms = []
                if l['type'] == 'centroid_connector' and u in centroid_set:
                    # Outgoing CC
                    for ol in outgoing[v]:
                        key = (u, v, ol['to'], s, t)
                        if key in y_vars:
                            outflow_terms.append(y_vars[key])
                else:
                    # Road link
                    for ol in outgoing[v]:
                        key = (u, v, ol['to'], s, t)
                        if key in y_vars:
                            outflow_terms.append(y_vars[key])
                
                if outflow_terms:
                    t_send = t - fftt + 1
                    nu_key = (u, v, s, t_send)
                    nd_key = (u, v, s, t)
                    if nu_key in NU_vars and nd_key in ND_vars:
                        m.addConstr(gp.quicksum(outflow_terms) <= NU_vars[nu_key] - ND_vars[nd_key])

    elapsed = time.time() - start_time
    print(f"Sending flow constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 5. Capacity constraints on road links (outgoing and incoming)
    for l in road_links:
        u, v = l['from'], l['to']
        # Convert capacity from vph to vehicles per time step
        time_step_hours = tp['time_step_duration_seconds'] / 3600.0
        cap = l['capacity_vph'] * time_step_hours
        
        dests_l = get_link_dests(l)
        
        for t in range(T):
            # Outgoing flow capacity
            outflow_all = []
            for s in dests_l:
                for ol in outgoing[v]:
                    key = (u, v, ol['to'], s, t)
                    if key in y_vars:
                        outflow_all.append(y_vars[key])
            if outflow_all:
                m.addConstr(gp.quicksum(outflow_all) <= cap)
            
            # Incoming flow capacity
            inflow_all = []
            for s in dests_l:
                for il in incoming[u]:
                    key = (il['from'], u, v, s, t)
                    if key in y_vars:
                        inflow_all.append(y_vars[key])
            if inflow_all:
                m.addConstr(gp.quicksum(inflow_all) <= cap)

    elapsed = time.time() - start_time
    print(f"Capacity constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 6. Congested wave constraints on road links
    for l in road_links:
        u, v = l['from'], l['to']
        cwtt = l['congested_travel_time_steps']
        n_jam = l['jam_density_vehicles']
        dests_l = get_link_dests(l)
        time_step_hours = tp['time_step_duration_seconds'] / 3600.0
        
        for t in range(T):
            inflow_all = []
            receiving_terms = []
            
            for s in dests_l:
                for il in incoming[u]:
                    key = (il['from'], u, v, s, t)
                    if key in y_vars:
                        inflow_all.append(y_vars[key])
                
                t_cw = t - cwtt + 1
                nd_key = (u, v, s, t_cw) if t_cw >= 0 else None
                nu_key = (u, v, s, t)
                
                if nd_key and nd_key in ND_vars and nu_key in NU_vars:
                    receiving_terms.append(ND_vars[nd_key] - NU_vars[nu_key])
                elif nu_key in NU_vars:
                    # t_cw < 0 means N_D(t_cw) = 0
                    receiving_terms.append(-NU_vars[nu_key])
            
            if inflow_all:
                if receiving_terms:
                    m.addConstr(gp.quicksum(inflow_all) <= gp.quicksum(receiving_terms) + n_jam)
                else:
                    m.addConstr(gp.quicksum(inflow_all) <= n_jam)

    elapsed = time.time() - start_time
    print(f"Congested wave constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 7. Parked vehicle constraints
    for c in centroids:
        # Initial
        m.addConstr(p_vars[(c, 0)] == initial_parked[c])
        
        for t in range(T):
            # Arrivals: vehicles arriving at c from incoming CCs
            arrival_terms = []
            for l in incoming_cc_by_centroid[c]:
                j = l['from']
                nu_key = (j, c, c, t)
                nd_key = (j, c, c, t)
                if nu_key in NU_vars and nd_key in ND_vars:
                    arrival_terms.append(NU_vars[nu_key] - ND_vars[nd_key])
            
            # Departures: vehicles leaving c via outgoing CCs
            departure_terms = []
            for l in outgoing_cc_by_centroid[c]:
                j = l['to']
                for s in get_link_dests(l):
                    key = (c, j, s, t)
                    if key in yc_vars:
                        departure_terms.append(yc_vars[key])
            
            rhs = p_vars[(c, t)]
            if arrival_terms:
                rhs = rhs + gp.quicksum(arrival_terms)
            if departure_terms:
                rhs = rhs - gp.quicksum(departure_terms)
            m.addConstr(p_vars[(c, t + 1)] == rhs)
        
        # Departure <= parked constraint
        for t in range(T):
            departure_terms = []
            for l in outgoing_cc_by_centroid[c]:
                j = l['to']
                for s in get_link_dests(l):
                    key = (c, j, s, t)
                    if key in yc_vars:
                        departure_terms.append(yc_vars[key])
            if departure_terms:
                m.addConstr(gp.quicksum(departure_terms) <= p_vars[(c, t)])

    # Fleet conservation at final time
    total_initial = sum(initial_parked[c] for c in centroids)
    m.addConstr(gp.quicksum(p_vars[(c, T)] for c in centroids) == total_initial)

    elapsed = time.time() - start_time
    print(f"Parked vehicle constraints added, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 8. Waiting traveler constraints
    for (o, d) in od_pairs:
        # Initial waiting at t=0
        d0 = demand_data[(o, d)].get(0, 0.0)
        m.addConstr(omega_vars[(o, d, 0)] == d0 - e_vars[(o, d, 0)])
        # e <= demand at t=0
        m.addConstr(e_vars[(o, d, 0)] <= d0)
        
        for t in range(1, T):
            new_demand = demand_data[(o, d)].get(t, 0.0)
            m.addConstr(omega_vars[(o, d, t)] == omega_vars[(o, d, t - 1)] + new_demand - e_vars[(o, d, t)])
            # e <= waiting
            m.addConstr(e_vars[(o, d, t)] <= omega_vars[(o, d, t - 1)] + new_demand)
        
        # At final time T:
        if T > 0:
            # omega at T should account for demand at T (if any) but problem says t in 0..T
            # Actually, let me re-read: "the waiting count at the next time step equals the current 
            # waiting count plus newly arriving demand minus travelers picked up at that time step"
            # omega(t+1) = omega(t) + demand(t+1) - e(t+1)
            # Wait, I need to be more careful. Let me re-read:
            # "the waiting count at the next time step equals the current waiting count plus newly 
            # arriving demand minus travelers picked up at that time step"
            # omega(o,d,t+1) = omega(o,d,t) + demand(o,d,t) - e(o,d,t) ? Or:
            # omega(o,d,t+1) = omega(o,d,t) + demand(o,d,t+1) - e(o,d,t+1) ?
            #
            # Hmm, "at that time step" - which time step? The "next time step" (t+1)?
            # "newly arriving demand" at step t or step t+1?
            #
            # I think the most natural reading is:
            # omega(t+1) = omega(t) + demand_arriving_at_t - pickup_at_t
            # where demand_arriving_at_t is the demand with departure time t.
            # But wait, the omega indexing and e indexing...
            #
            # Actually: "Unserviced traveler demand waiting at each origin for each destination 
            # evolves over time: the waiting count at the next time step equals the current waiting 
            # count plus newly arriving demand minus travelers picked up at that time step."
            #
            # "that time step" = current time step t.
            # omega(t+1) = omega(t) + demand(t) - e(t)
            # where demand(t) is demand arriving at time t and e(t) is pickup at time t.
            #
            # At t=0: omega(0) should be 0 (no one waiting before time 0).
            # omega(1) = omega(0) + demand(0) - e(0) = demand(0) - e(0)
            #
            # But earlier I wrote omega(0) = demand(0) - e(0) which assumed omega(0) includes 
            # demand at time 0 minus pickup at time 0. That's wrong if omega(0) = 0.
            #
            # Let me redo:
            # omega(0) = 0 (initially no one waiting)
            # omega(t+1) = omega(t) + demand(t) - e(t), for t = 0, ..., T-1
            # e(t) <= omega(t) + demand(t)  [can't pick up more than waiting + newly arrived]
            # e(t) <= departures from o for d at time t
            # omega(T) = 0 [all served]
            pass

    # Redo waiting constraints properly
    # First remove old omega and e constraints by rebuilding
    # Actually, I can't easily remove constraints from a Gurobi model. Let me fix the code above.
    # I'll just build the model fresh with correct constraints.
    
    # Instead of rebuilding, let me just recreate the model. The code above already has issues.
    # Let me restructure into a clean implementation.
    
    m.reset()
    m = gp.Model("SAV_v2")
    m.Params.TimeLimit = max(1, time_limit - 30)
    m.Params.Threads = 1
    m.Params.Method = 2
    m.Params.OutputFlag = 1
    
    elapsed = time.time() - start_time
    print(f"Rebuilding model cleanly... elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    time_step_hours = tp['time_step_duration_seconds'] / 3600.0

    # -- Variables --
    y_vars = {}
    for (u, v, w, il, ol) in all_turns:
        fftt_il = il['free_flow_travel_time_steps']
        for s in turn_dests[(u, v, w)]:
            for t in range(fftt_il, T):
                y_vars[(u, v, w, s, t)] = m.addVar(lb=0)

    yc_vars = {}
    for c in centroids:
        for l in outgoing_cc_by_centroid[c]:
            j = l['to']
            dests_l = get_link_dests(l)
            for s in dests_l:
                for t in range(T):
                    yc_vars[(c, j, s, t)] = m.addVar(lb=0)

    NU_vars = {}
    ND_vars = {}
    for l in links_list:
        u, v = l['from'], l['to']
        dests_l = get_link_dests(l)
        for s in dests_l:
            for t in range(T + 1):
                NU_vars[(u, v, s, t)] = m.addVar(lb=0)
                ND_vars[(u, v, s, t)] = m.addVar(lb=0)

    p_vars = {}
    for c in centroids:
        for t in range(T + 1):
            p_vars[(c, t)] = m.addVar(lb=0)

    e_vars = {}
    for (o, d) in od_pairs:
        for t in range(T):
            e_vars[(o, d, t)] = m.addVar(lb=0)

    omega_vars = {}
    for (o, d) in od_pairs:
        for t in range(T + 1):
            omega_vars[(o, d, t)] = m.addVar(lb=0)

    m.update()
    
    elapsed = time.time() - start_time
    print(f"Variables created: {m.NumVars}, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # -- Constraints --

    # Helper to get inflow terms for a link at time t for dest s
    def get_inflow(l, s, t):
        u, v = l['from'], l['to']
        terms = []
        if l['type'] == 'centroid_connector' and u in centroid_set:
            # Outgoing CC: inflow from centroid
            key = (u, v, s, t)
            if key in yc_vars:
                terms.append(yc_vars[key])
        else:
            # Road link or incoming CC: inflow from predecessor turns
            for il in incoming[u]:
                key = (il['from'], u, v, s, t)
                if key in y_vars:
                    terms.append(y_vars[key])
        return terms

    def get_outflow(l, s, t):
        u, v = l['from'], l['to']
        terms = []
        if l['type'] == 'centroid_connector' and v in centroid_set:
            # Incoming CC: no explicit outflow (handled by special rule)
            return terms
        for ol in outgoing[v]:
            key = (u, v, ol['to'], s, t)
            if key in y_vars:
                terms.append(y_vars[key])
        return terms

    # 1. Initial cumulative counts = 0
    for l in links_list:
        u, v = l['from'], l['to']
        for s in get_link_dests(l):
            m.addConstr(NU_vars[(u, v, s, 0)] == 0)
            m.addConstr(ND_vars[(u, v, s, 0)] == 0)

    # 2. N_U (entries) update
    for l in links_list:
        u, v = l['from'], l['to']
        for s in get_link_dests(l):
            for t in range(T):
                terms = get_inflow(l, s, t)
                if terms:
                    m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)] + gp.quicksum(terms))
                else:
                    m.addConstr(NU_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])

    elapsed = time.time() - start_time
    print(f"N_U update constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 3. N_D (exits) update
    for l in links_list:
        u, v = l['from'], l['to']
        for s in get_link_dests(l):
            if l['type'] == 'centroid_connector' and v in centroid_set:
                # Incoming CC special rule
                for t in range(T):
                    m.addConstr(ND_vars[(u, v, s, t + 1)] == NU_vars[(u, v, s, t)])
            else:
                for t in range(T):
                    terms = get_outflow(l, s, t)
                    if terms:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)] + gp.quicksum(terms))
                    else:
                        m.addConstr(ND_vars[(u, v, s, t + 1)] == ND_vars[(u, v, s, t)])

    elapsed = time.time() - start_time
    print(f"N_D update constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 4. Sending flow constraints (for road links and outgoing CCs)
    for l in links_list:
        u, v = l['from'], l['to']
        if l['type'] == 'centroid_connector' and v in centroid_set:
            continue  # incoming CC: no outflow
        
        fftt = l['free_flow_travel_time_steps']
        for s in get_link_dests(l):
            for t in range(fftt, T):
                terms = get_outflow(l, s, t)
                if terms:
                    t_send = t - fftt + 1
                    m.addConstr(gp.quicksum(terms) <= NU_vars[(u, v, s, t_send)] - ND_vars[(u, v, s, t)])

    elapsed = time.time() - start_time
    print(f"Sending flow constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 5. Capacity constraints on road links
    for l in road_links:
        u, v = l['from'], l['to']
        cap = l['capacity_vph'] * time_step_hours
        dests_l = get_link_dests(l)
        
        for t in range(T):
            # Outgoing capacity
            out_terms = []
            for s in dests_l:
                out_terms.extend(get_outflow(l, s, t))
            if out_terms:
                m.addConstr(gp.quicksum(out_terms) <= cap)
            
            # Incoming capacity
            in_terms = []
            for s in dests_l:
                in_terms.extend(get_inflow(l, s, t))
            if in_terms:
                m.addConstr(gp.quicksum(in_terms) <= cap)

    elapsed = time.time() - start_time
    print(f"Capacity constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 6. Congested wave constraints on road links
    for l in road_links:
        u, v = l['from'], l['to']
        cwtt = l['congested_travel_time_steps']
        n_jam = l['jam_density_vehicles']
        dests_l = get_link_dests(l)
        
        for t in range(T):
            in_terms = []
            recv_terms = []
            
            for s in dests_l:
                in_terms.extend(get_inflow(l, s, t))
                
                t_cw = t - cwtt + 1
                if t_cw >= 0:
                    recv_terms.append(ND_vars[(u, v, s, t_cw)] - NU_vars[(u, v, s, t)])
                else:
                    recv_terms.append(-NU_vars[(u, v, s, t)])
            
            if in_terms:
                m.addConstr(gp.quicksum(in_terms) <= gp.quicksum(recv_terms) + n_jam)

    elapsed = time.time() - start_time
    print(f"Congested wave constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 7. Parked vehicle constraints
    for c in centroids:
        m.addConstr(p_vars[(c, 0)] == initial_parked[c])
        
        for t in range(T):
            # Arrivals: flow exiting incoming CCs at time t
            # = N_U(incoming_CC, c, t) - N_D(incoming_CC, c, t) [vehicles on CC at time t]
            arrival_terms = []
            for l in incoming_cc_by_centroid[c]:
                j = l['from']
                if (j, c, c, t) in NU_vars and (j, c, c, t) in ND_vars:
                    arrival_terms.append(NU_vars[(j, c, c, t)] - ND_vars[(j, c, c, t)])
            
            # Departures
            dep_terms = []
            for l in outgoing_cc_by_centroid[c]:
                j = l['to']
                for s in get_link_dests(l):
                    if (c, j, s, t) in yc_vars:
                        dep_terms.append(yc_vars[(c, j, s, t)])
            
            rhs = p_vars[(c, t)]
            if arrival_terms:
                rhs = rhs + gp.quicksum(arrival_terms)
            if dep_terms:
                rhs = rhs - gp.quicksum(dep_terms)
            m.addConstr(p_vars[(c, t + 1)] == rhs)
            
            # Departure <= parked
            if dep_terms:
                m.addConstr(gp.quicksum(dep_terms) <= p_vars[(c, t)])

    # Fleet conservation
    total_initial = sum(initial_parked[c] for c in centroids)
    m.addConstr(gp.quicksum(p_vars[(c, T)] for c in centroids) == total_initial)

    elapsed = time.time() - start_time
    print(f"Parked constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # 8. Waiting traveler constraints
    # omega(0) = 0 initially
    # omega(t+1) = omega(t) + demand(t) - e(t) for t = 0, ..., T-1
    # But wait, we need omega at t=0 to be 0, and e at t=0 to handle demand at t=0.
    # Actually, let me think again. omega(o,d,t) = number waiting at time t.
    # At time 0, demand(0) arrives and e(0) are picked up.
    # omega(0) should be the waiting BEFORE pickups. Or AFTER?
    # 
    # The constraint says:
    # "the waiting count at the next time step equals the current waiting count plus newly arriving 
    # demand minus travelers picked up at that time step"
    # omega(t+1) = omega(t) + demand(t) - e(t)
    # 
    # "The number of travelers picked up at an origin for a given destination at any time step 
    # cannot exceed the number of travelers currently waiting there for that destination"
    # e(t) <= omega(t) + demand(t)  
    # Wait, it says "currently waiting" which is omega(t). But demand(t) just arrived.
    # Hmm, maybe e(t) <= omega(t)? But if omega(0) = 0, then e(0) = 0 and demand at t=0 
    # isn't served until t=1? That seems wrong.
    #
    # Let me re-read: "cannot exceed the number of travelers currently waiting there for that destination"
    # "currently waiting" at time t includes those who arrived at t?
    # I think omega(t) already includes demand arriving at t. So:
    # omega(0) = demand(0) (people arriving at t=0 are "waiting")
    # omega(t+1) = omega(t) - e(t) + demand(t+1)
    # e(t) <= omega(t)
    #
    # Wait but the formula says omega(t+1) = omega(t) + demand(t) - e(t). Not demand(t+1).
    # Let me try: omega represents waiting AFTER all operations at time t.
    # At t=0: demand(0) arrives, e(0) picked up. omega(0) = demand(0) - e(0).
    # omega(1) = omega(0) + demand(1) - e(1) = demand(0) - e(0) + demand(1) - e(1)
    # e(t) <= omega(t-1) + demand(t) = waiting before pickup at t
    # But omega(-1) doesn't exist.
    # For t=0: e(0) <= demand(0)
    # For t>0: e(t) <= omega(t-1) + demand(t)
    #
    # Hmm, this is getting confusing. Let me use a cleaner formulation.
    # 
    # Let W(t) be the number waiting at the beginning of time step t (before pickup and new arrivals).
    # W(0) = 0
    # Available at time t = W(t) + demand(t)
    # e(t) <= Available(t)
    # W(t+1) = Available(t) - e(t) = W(t) + demand(t) - e(t)
    #
    # omega(t) as output = W(t) + demand(t) - e(t) = W(t+1) = waiting at END of time step t
    # Or omega(t) = W(t) = waiting at START of time step t
    #
    # The problem says omega is indexed 0..T.
    # "the unserviced count for every origin-destination pair at the final time step must be zero"
    # omega(T) = 0
    #
    # And the objective includes "total traveler waiting time = sum over all OD pairs and time steps 
    # the unserviced waiting demand" = sum_{o,d,t} omega(o,d,t)
    #
    # If omega(t) = W(t+1) = waiting after step t, then summing omega gives the total time-waiting.
    # omega(t) = omega(t-1) + demand(t) - e(t), omega(-1) = 0, so omega(0) = demand(0) - e(0).
    # omega(T-1) should be 0 if omega(T) is defined by the update as:
    # omega(T) = omega(T-1) + demand(T) - e(T)
    # But e is only defined for t in 0..T-1 and omega for 0..T.
    #
    # Hmm, demand period might end before T. Let me just look at the update:
    # omega(o,d,t+1) = omega(o,d,t) + demand(o,d,t) - e(o,d,t) for t = 0,...,T-1
    # Wait, but e is defined for t = 0,...,T-1 (T time steps from the e_vars definition).
    # omega defined for t = 0,...,T.
    # 
    # Let me use: omega(t) = waiting count at the START of step t = W(t).
    # omega(0) = 0
    # omega(t+1) = omega(t) + demand_arriving_at_t - e(t)
    # e(t) <= omega(t) + demand_arriving_at_t  [can pick up waiting + newly arrived]
    # Wait, the problem says: "cannot exceed the number of travelers currently waiting"
    # If "currently waiting" means omega(t) (start of step, before new arrivals), then:
    # e(t) <= omega(t)
    # But then demand arriving at t=0 can't be served until t=1, since omega(0) = 0.
    #
    # But! "also cannot exceed the total number of vehicles departing that origin toward that 
    # destination at that time step"
    # e(o,d,t) <= sum_j yc(o,j,d,t) [vehicles leaving o for d at t]
    #
    # I think the intended interpretation is:
    # omega(t) includes the newly arrived demand. So:
    # omega(0) = demand(0)  [waiting before any pickup at t=0]
    # e(0) <= omega(0) = demand(0)
    # omega(1) = omega(0) - e(0) + demand(1)
    # General: omega(t+1) = omega(t) - e(t) + demand(t+1)
    #
    # The problem's formula: omega(t+1) = omega(t) + "newly arriving demand" - e(t)
    # If "newly arriving demand" at step t is demand(t), then:
    # omega(t+1) = omega(t) + demand(t) - e(t)
    # With omega(0) = 0: omega(1) = demand(0) - e(0)
    # e(0) <= omega(0) = 0 means no pickup at t=0 -> demand(0) waits until t=1.
    #
    # With omega(0) = demand(0): omega(1) = demand(0) + demand(0) - e(0) = 2*demand(0) - e(0)
    # That double counts.
    #
    # I think the correct version is:
    # omega(t+1) = omega(t) + demand(t) - e(t), starting with omega(0) = 0
    # e(t) <= omega(t) + demand(t)  [can serve the newly arrived too]
    # "currently waiting" = omega(t) + demand(t) = total available
    #
    # Or: e(t) <= omega(t) meaning you can only serve those who were already waiting (not new arrivals).
    #
    # I'll go with: e(t) <= omega(t) + demand(t) and omega(t+1) = omega(t) + demand(t) - e(t),
    # starting with omega(0) = 0. But the problem statement says e(t) <= "currently waiting" 
    # which I'll interpret as omega(t) + demand(t) (total available at time t).
    # Actually wait, "currently waiting" naturally means those already waiting, i.e., omega(t).
    # But then at t=0 nobody can be picked up.
    #
    # Let me re-read one more time: "the waiting count at the next time step equals the current 
    # waiting count plus newly arriving demand minus travelers picked up at that time step"
    # omega(t+1) = omega(t) + demand_new(t) - e(t)
    #
    # "The number of travelers picked up at an origin for a given destination at any time step 
    # cannot exceed the number of travelers currently waiting there for that destination"
    # e(t) <= omega(t)
    #
    # With omega(0) = 0: e(0) <= 0, so e(0) = 0, omega(1) = demand(0).
    # e(1) <= omega(1) = demand(0). So travelers arriving at t=0 can first be picked up at t=1.
    # This means they wait 1 time step minimum. For an SAV system this might make sense: 
    # the traveler arrives and waits, then at the next step a vehicle picks them up.
    #
    # The objective counts sum omega(t) for t = 0..T. omega(0) = 0 always.
    # Let me also check: "sum over all OD pairs and all time steps the unserviced waiting demand"
    # = sum_{o,d} sum_{t=0}^{T} omega(o,d,t)
    
    # OK let me just go with this interpretation:
    # omega(0) = 0
    # omega(t+1) = omega(t) + demand(t) - e(t) for t = 0,...,T-1
    # e(t) <= omega(t) [for the "waiting" constraint]
    # Also: e(t) <= omega(t) + demand(t) is implied since demand(t) >= 0
    # Actually no: the problem says e(t) <= omega(t), which is stricter.
    # But then demand arriving at t can't be served until t+1. Let me check if that's consistent.
    # omega(T) = 0 requires all demand served.
    
    # Hmm, but actually that interpretation might not match well with "e(t) also cannot exceed the 
    # total vehicles departing". In this interpretation, at t=0 no one can be picked up regardless 
    # of vehicles available. The pickup happens at t=1 at earliest.
    
    # Let me just go with this and see if it produces feasible solutions.
    # If e(t) <= omega(t) and omega(0) = 0, then effectively demand at time t is first serviceable 
    # at time t+1. omega(1) = demand(0), then e(1) <= omega(1) = demand(0).
    
    # Actually, I just realized there's ambiguity about what "demand(t)" means in the update.
    # The problem says: "Person-trip demand is specified for each origin-destination centroid pair 
    # at each time step, giving the number of travelers wishing to depart from a given origin to 
    # a given destination at that time."
    # 
    # So demand(o,d,t) is travelers wanting to depart at time t.
    # "newly arriving demand" at step t in the omega update = demand(o,d,t).
    #
    # Hmm wait, let me reconsider. Maybe omega(t) represents the state AFTER demand arrives but 
    # BEFORE pickup. Then:
    # At the start of step t: omega_before = omega(t-1) [carried over from previous step]
    # Demand arrives: available = omega_before + demand(t) 
    # But that's not omega(t). 
    #
    # Or maybe the update is:
    # omega(t) = omega(t-1) + demand(t) - e(t)
    # where omega(t) is the waiting count at the END of step t (or equivalently, start of step t+1).
    # e(t) is pickup during step t.
    # demand(t) arrives during step t.
    # e(t) <= omega(t-1) + demand(t) = available during step t
    #
    # omega(0) = 0 + demand(0) - e(0) = demand(0) - e(0) [no one waiting before, demand(0) arrives]
    # e(0) <= 0 + demand(0) = demand(0) [can serve demand(0) immediately]
    # This makes more sense!
    #
    # But then we need omega(-1) = 0 or equivalently, the formula omega(t) = omega(t-1) + demand(t) - e(t)
    # with omega(-1) = 0, running for t = 0, 1, ..., T-1 (since e is defined for t = 0..T-1).
    # omega(T-1) = 0 means all served by T-1.
    # But the problem says omega(T) = 0. And omega is defined for t = 0..T.
    #
    # Actually wait, re-reading the problem: "the waiting count at the next time step equals the 
    # current waiting count plus newly arriving demand minus travelers picked up at that time step"
    # 
    # Let me map: "current" = t, "next" = t+1:
    # omega(t+1) = omega(t) + demand_arriving_at_t+1... no.
    # "newly arriving demand" = demand arriving at "that time step" = the current time step t? 
    # Or the next time step t+1?
    # "minus travelers picked up at that time step" = e(t)
    # "that time step" seems to refer to the same time step as "current".
    #
    # So: omega(t+1) = omega(t) + demand(t) - e(t)
    # With omega(0) given as initial condition.
    # "the unserviced count for every OD pair at the final time step must be zero" -> omega(T) = 0
    # "sum over all time steps the unserviced waiting demand" -> sum_{t=0}^{T} omega(t)
    #
    # If omega(0) = 0 and e(t) <= omega(t):
    # At t=0: e(0) <= omega(0) = 0, so e(0) = 0.
    # omega(1) = 0 + demand(0) - 0 = demand(0)
    # At t=1: e(1) <= omega(1) = demand(0). Can serve demand(0) now.
    # omega(2) = demand(0) - e(1) + demand(1)
    # Waiting time for demand(0) is at least 1 step (they're counted in omega(1)).
    # Total waiting includes omega(0) = 0, omega(1) = demand(0), etc.
    # sum omega = 0 + demand(0) + (demand(0) - e(1) + demand(1)) + ...
    
    # Alternatively, if omega(0) includes demand(0):
    # omega(0) = demand(0)  [initial state has demand(0) waiting]
    # But then the update omega(1) = omega(0) + demand(0) - e(0) = 2*demand(0) - e(0) which double counts.
    
    # UNLESS the update is: omega(t+1) = omega(t) + demand(t+1) - e(t)
    # omega(0) = demand(0)
    # e(0) <= omega(0) = demand(0) [can serve immediately]
    # omega(1) = demand(0) - e(0) + demand(1) [leftover from t=0 plus new arrivals at t=1]
    # This makes sense! And e(t) <= omega(t) works since omega(t) includes current demand.
    #
    # The problem says "plus newly arriving demand" which in this case would be demand(t+1), 
    # the demand arriving at the next time step. Hmm, that's an unusual phrasing.
    #
    # I think the cleanest interpretation that makes both the update and the constraint work is:
    # omega(t) = total unserviced travelers at time t (includes those who just arrived at t)
    # e(t) = pickups at time t
    # e(t) <= omega(t)  [can only serve those who are there]
    # e(t) <= vehicles departing at t
    # omega(t+1) = omega(t) - e(t) + demand(t+1)  [remaining after pickup, plus new arrivals]
    #
    # Let me verify with the problem's wording:
    # "the waiting count at the NEXT time step" = omega(t+1)
    # "equals the CURRENT waiting count" = omega(t)
    # "plus newly arriving demand" = demand arriving at the next step = demand(t+1)
    # "minus travelers picked up at THAT time step" = e at the current step = e(t)
    #
    # omega(t+1) = omega(t) + demand(t+1) - e(t)
    # Hmm, but "at that time step" is ambiguous. Is it the current step t or the next step t+1?
    # If "that time step" = t+1: omega(t+1) = omega(t) + demand(t+1) - e(t+1)
    # Then e(t+1) <= omega(t) + demand(t+1) [total available at t+1]
    # But the constraint says e(t) <= omega(t) [currently waiting at t].
    # If we use e(t+1) <= omega(t+1)... but omega(t+1) depends on e(t+1). Circular.
    # 
    # I think the problem means:
    # omega(t+1) = omega(t) + demand(t+1) - e(t+1)
    # e(t) <= omega(t) [travelers currently waiting, which includes demand(t)]
    # omega(0) = demand(0) [initial waiting = initial demand at t=0]
    # e(0) <= omega(0) = demand(0) [can serve demand at t=0]
    # omega(T) = 0 [all served]
    #
    # Let me verify: omega(1) = omega(0) + demand(1) - e(1) = demand(0) - e(0) + demand(1) - e(1)
    # Wait no: omega(1) = omega(0) + demand(1) - e(1)
    # But e(0) isn't in the equation for omega(1)!
    # That means travelers picked up at t=0 don't reduce omega(1)?
    #
    # OK this is clearly wrong. Let me try:
    # omega(t+1) = omega(t) - e(t) + demand(t+1)
    # Same as omega(t+1) = omega(t) + demand(t+1) - e(t)
    # Pickup e(t) happens at step t, reducing omega. New demand(t+1) arrives at step t+1.
    # omega(0) = demand(0). e(0) <= omega(0).
    # omega(1) = omega(0) - e(0) + demand(1) = demand(0) - e(0) + demand(1). OK.
    # omega(T) = 0 means all served by time T.
    #
    # But this has "demand(t+1)" appearing in the formula, which means demand at time T would be 
    # demand(T). If the demand period ends before T, demand(T) = 0 and it's fine.
    # Also, we need e defined for t=0..T-1. omega defined for t=0..T.
    # The last update: omega(T) = omega(T-1) - e(T-1) + demand(T)
    #
    # This seems workable! Let me also check: the sum of omega in the objective = sum_{t=0}^T omega(t).
    # omega(0) = demand(0), so even if all demand is served immediately, omega(0) > 0 and contributes 
    # to the objective. That represents the demand being present (waiting) at time 0.
    
    # Actually, wait. If demand(0) > 0 and we serve all of it at t=0 (e(0) = demand(0)), then:
    # omega(0) = demand(0), omega(1) = demand(0) - demand(0) + demand(1) = demand(1).
    # The objective counts omega(0) = demand(0) even though they were served immediately.
    # That means every traveler contributes at least 1 to the objective just for being present.
    # This might be the intended behavior (minimum waiting of 1 step).
    
    # Alternatively: omega(t) represents waiting BEFORE demand(t) arrives:
    # omega(0) = 0
    # demand(t) arrives, then e(t) picked up from omega(t) + demand(t).
    # omega(t+1) = omega(t) + demand(t) - e(t)
    # e(t) <= omega(t) + demand(t)
    #
    # The problem says e(t) <= omega(t) [currently waiting]. If omega(t) doesn't include demand(t):
    # e(0) <= omega(0) = 0. Can't serve at t=0. Minimum wait of 1 step.
    # omega(1) = 0 + demand(0) - 0 = demand(0). e(1) <= demand(0).
    # If served at t=1: omega(2) = demand(0) - demand(0) + demand(1) = demand(1).
    # 
    # But then the constraint "e(t) <= omega(t)" means e(t) <= omega(t) = accumulated from previous steps.
    # New demand at step t can't be served until step t+1.
    # Objective: sum omega = 0 + demand(0) + (demand(0)-e(1)+demand(1)) + ...
    # Even if served at t=1, demand(0) contributes demand(0) to objective (in omega(1)).
    # Minimum waiting time per traveler arriving at t: 1 step.
    
    # I think this might be the intended interpretation given the problem says "e(t) <= omega(t)" 
    # (can't serve newly arrived in same step).
    
    # Let me go with: omega(0) = 0, omega(t+1) = omega(t) + demand(t) - e(t), e(t) <= omega(t).
    # Note: e(t) <= omega(t) means at t=0, e(0) = 0 always. So I won't even bother with e at t=0.
    # Actually, e(0) <= omega(0) = 0 means e(0) = 0. omega(1) = demand(0).
    
    # Hmm, but I realize there's one more constraint I haven't considered:
    # "e(t) also cannot exceed the total number of vehicles departing that origin toward that 
    # destination at that time step"
    # e(o,d,t) <= sum_j yc(o,j,d,t)
    # A vehicle departing o for d carries one traveler. So pickups are limited by vehicles.
    
    # OK let me just implement the simpler interpretation and see.

    for (o, d) in od_pairs:
        m.addConstr(omega_vars[(o, d, 0)] == 0)
        
        for t in range(T):
            dem_t = demand_data[(o, d)].get(t, 0.0)
            m.addConstr(omega_vars[(o, d, t + 1)] == omega_vars[(o, d, t)] + dem_t - e_vars[(o, d, t)])
            m.addConstr(e_vars[(o, d, t)] <= omega_vars[(o, d, t)] + dem_t)
            
            # e <= vehicles departing
            veh_terms = []
            for l in outgoing_cc_by_centroid[o]:
                j = l['to']
                if (o, j, d, t) in yc_vars:
                    veh_terms.append(yc_vars[(o, j, d, t)])
            if veh_terms:
                m.addConstr(e_vars[(o, d, t)] <= gp.quicksum(veh_terms))
            else:
                m.addConstr(e_vars[(o, d, t)] == 0)
        
        # omega(T) = 0
        m.addConstr(omega_vars[(o, d, T)] == 0)

    elapsed = time.time() - start_time
    print(f"Waiting constraints, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # No turning flow before fftt (already enforced by variable creation range)
    
    # All links should be empty at time T (optional but good for feasibility)
    # Actually, the problem doesn't require this. It only requires fleet conservation.
    # But vehicles in transit are still counted in the objective via cumulative counts.

    # OBJECTIVE
    # Total vehicle time = sum_{link, dest, time} (N_U - N_D)
    # Total waiting time = sum_{o,d,t} omega
    obj_terms = []
    
    for l in links_list:
        u, v = l['from'], l['to']
        for s in get_link_dests(l):
            for t in range(T + 1):
                obj_terms.append(NU_vars[(u, v, s, t)] - ND_vars[(u, v, s, t)])
    
    for (o, d) in od_pairs:
        for t in range(T + 1):
            obj_terms.append(omega_vars[(o, d, t)])
    
    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    
    m.update()
    elapsed = time.time() - start_time
    print(f"Model built: {m.NumVars} vars, {m.NumConstrs} constrs, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    # Solve
    def callback(model, where):
        if where == GRB.Callback.BARRIER:
            pass
        if where == GRB.Callback.MESSAGE:
            pass

    m.optimize()

    elapsed = time.time() - start_time
    print(f"Optimization done, status: {m.Status}, elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL or m.SolCount > 0:
        obj_val = m.ObjVal
        if logger:
            logger.log(obj_val)
        
        # Extract solution
        sol = {"objective_value": obj_val}
        
        # y
        y_out = {}
        for (u, v, w, s, t), var in y_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{u}|{v}|{w}|{s}|{t}"
                y_out[key] = val
        sol["y"] = y_out
        
        # y_centroid
        yc_out = {}
        for (c, j, s, t), var in yc_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{c}|{j}|{s}|{t}"
                yc_out[key] = val
        sol["y_centroid"] = yc_out
        
        # N_U
        nu_out = {}
        for (u, v, s, t), var in NU_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{u}|{v}|{s}|{t}"
                nu_out[key] = val
        sol["N_U"] = nu_out
        
        # N_D
        nd_out = {}
        for (u, v, s, t), var in ND_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{u}|{v}|{s}|{t}"
                nd_out[key] = val
        sol["N_D"] = nd_out
        
        # p
        p_out = {}
        for (c, t), var in p_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{c}|{t}"
                p_out[key] = val
        sol["p"] = p_out
        
        # e
        e_out = {}
        for (o, d, t), var in e_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{o}|{d}|{t}"
                e_out[key] = val
        sol["e"] = e_out
        
        # omega
        om_out = {}
        for (o, d, t), var in omega_vars.items():
            val = var.X
            if abs(val) > 1e-10:
                key = f"{o}|{d}|{t}"
                om_out[key] = val
        sol["omega"] = om_out
        
        with open(solution_path, 'w') as f:
            json.dump(sol, f)
        
        print(f"Solution written, objective: {obj_val}")
    else:
        print(f"No feasible solution found, status: {m.Status}")
        # Write empty solution
        sol = {
            "objective_value": float('inf'),
            "y": {},
            "y_centroid": {},
            "N_U": {},
            "N_D": {},
            "p": {},
            "e": {},
            "omega": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(sol, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)