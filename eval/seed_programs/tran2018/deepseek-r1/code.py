import bisect
from collections import defaultdict

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it)); m = int(next(it)); R = int(next(it))
    
    dist = [[10**9] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        
    for _ in range(m):
        u = int(next(it)); v = int(next(it)); d = int(next(it))
        if d < dist[u][v]:
            dist[u][v] = d
            dist[v][u] = d
            
    num_od_pairs = int(next(it))
    od_pairs = []
    for _ in range(num_od_pairs):
        o = int(next(it)); d = int(next(it)); flow_od = int(next(it)); num_nodes = int(next(it))
        forward_path = [int(next(it)) for _ in range(num_nodes)]
        od_pairs.append((o, d, flow_od, forward_path))
        
    coverage = defaultdict(set)
    critical_arcs = []
    flows = [0] * num_od_pairs
    
    for od_idx, (o, d, flow_od, forward_path) in enumerate(od_pairs):
        flows[od_idx] = flow_od
        backward_path = forward_path[-2::-1]
        journey = forward_path + backward_path
        L = len(journey)
        D_total = [0] * L
        for i in range(L-1):
            u = journey[i]
            v = journey[i+1]
            d_uv = dist[u][v]
            D_total[i+1] = D_total[i] + d_uv
            
        for i in range(L-1):
            if D_total[i+1] > R:
                arc_id = (od_idx, i)
                critical_arcs.append(arc_id)
                F = D_total[i+1] - R
                j0 = bisect.bisect_left(D_total, F, 0, i+1)
                candidate_set = set()
                for j in range(j0, i+1):
                    candidate_set.add(journey[j])
                for node in candidate_set:
                    coverage[node].add(arc_id)
                    
    count_od = [0] * num_od_pairs
    for arc in critical_arcs:
        od_idx = arc[0]
        count_od[od_idx] += 1
        
    base_flow = 0
    for od_idx in range(num_od_pairs):
        if count_od[od_idx] == 0:
            base_flow += flows[od_idx]
            
    selected_stations = set()
    covered_arcs = set()
    total_flow_covered = base_flow
    
    for step in range(p):
        best_station = None
        best_gain = 0
        for node in coverage:
            if node in selected_stations:
                continue
            new_arcs = coverage[node] - covered_arcs
            if not new_arcs:
                continue
            od_arcs = defaultdict(set)
            for arc in new_arcs:
                od_idx = arc[0]
                od_arcs[od_idx].add(arc)
            gain = 0
            for od_idx, arcs_set in od_arcs.items():
                if count_od[od_idx] > 0 and count_od[od_idx] - len(arcs_set) == 0:
                    gain += flows[od_idx]
            if gain > best_gain:
                best_gain = gain
                best_station = node
                
        if best_gain == 0:
            break
            
        total_flow_covered += best_gain
        selected_stations.add(best_station)
        new_arcs = coverage[best_station] - covered_arcs
        covered_arcs |= new_arcs
        od_arcs = defaultdict(set)
        for arc in new_arcs:
            od_idx = arc[0]
            od_arcs[od_idx].add(arc)
        for od_idx, arcs_set in od_arcs.items():
            count_od[od_idx] -= len(arcs_set)
            
    print(total_flow_covered)

if __name__ == "__main__":
    main()