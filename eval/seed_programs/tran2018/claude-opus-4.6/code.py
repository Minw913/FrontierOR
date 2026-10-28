import argparse
import json
import time
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

    all_nodes = set(node['id'] for node in instance['nodes'])

    def preprocess_od_pair(od_idx):
        od = od_pairs[od_idx]
        path = od['shortest_path']
        flow = od['flow']

        if len(path) <= 1:
            return flow, set(), True, None

        round_trip = path + path[-2::-1]

        num_nodes_rt = len(round_trip)
        seg_dists = []
        for i in range(num_nodes_rt - 1):
            u, v = round_trip[i], round_trip[i + 1]
            d = arc_dist.get((u, v))
            if d is None:
                return flow, None, False, None
            seg_dists.append(d)

        for d in seg_dists:
            if d > R:
                return flow, None, False, None

        total_rt_dist = sum(seg_dists)
        if total_rt_dist <= R:
            return flow, set(), True, None

        path_nodes = set(path)
        return flow, path_nodes, None, (round_trip, seg_dists)

    always_covered = []
    infeasible = []
    needs_stations = []

    for idx in range(num_od_pairs):
        flow, candidates, auto_covered, rt_data = preprocess_od_pair(idx)
        if auto_covered is True:
            always_covered.append(idx)
        elif auto_covered is False and candidates is None:
            infeasible.append(idx)
        else:
            needs_stations.append((idx, flow, candidates, rt_data))

    def is_covered(rt_data, open_stations):
        round_trip, seg_dists = rt_data
        fuel = R
        for i in range(len(seg_dists)):
            if round_trip[i] in open_stations:
                fuel = R
            fuel -= seg_dists[i]
            if fuel < -1e-9:
                return False
        return True

    def compute_covered_flow(open_stations_set):
        total = sum(od_pairs[idx]['flow'] for idx in always_covered)
        covered_indices = list(always_covered)
        for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
            if is_covered(rt_data, open_stations_set):
                total += flow
                covered_indices.append(idx)
        return total, covered_indices

    all_candidates = set()
    for idx, flow, candidates, rt_data in needs_stations:
        all_candidates.update(candidates)

    all_candidates = sorted(all_candidates)
    all_candidates_set = set(all_candidates)

    always_covered_flow = sum(od_pairs[idx]['flow'] for idx in always_covered)
    best_objective = always_covered_flow
    best_stations_set = set()
    best_covered = list(always_covered)

    if logger:
        logger.log(best_objective)

    if p == 0:
        result = {
            "objective_value": best_objective,
            "stations": [],
            "refueled_od_pairs": sorted(best_covered)
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f)
        return

    # Greedy construction
    selected_stations = set()

    for step in range(p):
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.25:
            break

        best_gain = -1
        best_node = None

        for node in all_candidates:
            if node in selected_stations:
                continue
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
        else:
            # Pick any remaining candidate
            for node in all_candidates:
                if node not in selected_stations:
                    selected_stations.add(node)
                    break
            else:
                for node in all_nodes:
                    if node not in selected_stations:
                        selected_stations.add(node)
                        break

    # Fill remaining stations if greedy stopped early
    while len(selected_stations) < p:
        for node in all_candidates:
            if node not in selected_stations:
                selected_stations.add(node)
                if len(selected_stations) >= p:
                    break
        if len(selected_stations) < p:
            for node in all_nodes:
                if node not in selected_stations:
                    selected_stations.add(node)
                    if len(selected_stations) >= p:
                        break

    obj, covered = compute_covered_flow(selected_stations)
    if obj > best_objective:
        best_objective = obj
        best_stations_set = set(selected_stations)
        best_covered = covered
        if logger:
            logger.log(best_objective)

    # Local search: swap stations
    def local_search(current_stations_set, time_fraction):
        nonlocal best_objective, best_stations_set, best_covered

        current_set = set(current_stations_set)
        current_obj = compute_covered_flow(current_set)[0]

        improved = True
        while improved and (time.time() - start_time) < time_limit * time_fraction:
            improved = False
            sel_list = sorted(current_set)
            non_sel_list = sorted(all_candidates_set - current_set)

            for i in range(len(sel_list)):
                if (time.time() - start_time) > time_limit * time_fraction:
                    break
                found = False
                for j in range(len(non_sel_list)):
                    if (time.time() - start_time) > time_limit * time_fraction:
                        break
                    test_set = set(current_set)
                    test_set.discard(sel_list[i])
                    test_set.add(non_sel_list[j])

                    test_obj, test_covered = compute_covered_flow(test_set)
                    if test_obj > current_obj:
                        current_obj = test_obj
                        current_set = test_set
                        if current_obj > best_objective:
                            best_objective = current_obj
                            best_stations_set = set(current_set)
                            best_covered = test_covered
                            if logger:
                                logger.log(best_objective)
                        improved = True
                        found = True
                        break
                if found:
                    break

        return current_set

    selected_stations = local_search(selected_stations, 0.5)

    # Try MIP if time permits
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed

    if remaining_time > 5 and len(all_candidates) <= 500 and len(needs_stations) <= 2000:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            total_rt_nodes = sum(len(rt_data[0]) for _, _, _, rt_data in needs_stations)

            if total_rt_nodes <= 50000:
                model = gp.Model("FRLM")
                model.setParam('TimeLimit', max(remaining_time - 2, 1))
                model.setParam('Threads', 1)
                model.setParam('OutputFlag', 0)

                y = {}
                for node in all_candidates:
                    y[node] = model.addVar(vtype=GRB.BINARY, name=f"y_{node}")

                model.addConstr(gp.quicksum(y[node] for node in all_candidates) <= p)

                M_val = R * 2 + 1
                z = {}
                obj_expr = gp.LinExpr()

                for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                    round_trip, seg_dists = rt_data
                    K = len(round_trip)

                    z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")
                    obj_expr += flow * z[i]

                    fuel = {}
                    g = {}

                    for j in range(K):
                        fuel[j] = model.addVar(lb=-M_val, ub=R, name=f"fuel_{i}_{j}")
                        g[j] = model.addVar(lb=-M_val, ub=R, name=f"g_{i}_{j}")

                    model.addConstr(fuel[0] == R)
                    model.addConstr(g[0] == R)

                    for j in range(1, K):
                        node_j = round_trip[j]
                        model.addConstr(fuel[j] == g[j - 1] - seg_dists[j - 1])

                        if node_j in y:
                            model.addConstr(g[j] >= fuel[j])
                            model.addConstr(g[j] <= R)
                            model.addConstr(g[j] >= R - M_val * (1 - y[node_j]))
                            model.addConstr(g[j] <= fuel[j] + M_val * y[node_j])
                        else:
                            model.addConstr(g[j] == fuel[j])

                        model.addConstr(fuel[j] >= -M_val * (1 - z[i]))

                model.setObjective(obj_expr, GRB.MAXIMIZE)

                # Warm start
                for node in all_candidates:
                    y[node].Start = 1.0 if node in best_stations_set else 0.0

                for i, (idx, flow, candidates, rt_data) in enumerate(needs_stations):
                    z[i].Start = 1.0 if idx in set(best_covered) else 0.0

                model.optimize()

                if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] and model.SolCount > 0:
                    mip_stations = set(node for node in all_candidates if y[node].X > 0.5)
                    mip_obj, mip_covered = compute_covered_flow(mip_stations)

                    if mip_obj > best_objective:
                        best_objective = mip_obj
                        best_stations_set = mip_stations
                        best_covered = mip_covered
                        if logger:
                            logger.log(best_objective)
        except Exception:
            pass

    # Final local search
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.95:
        local_search(best_stations_set, 0.95)

    # Ensure exactly p stations
    final_stations = set(best_stations_set)
    while len(final_stations) < p:
        for node in all_nodes:
            if node not in final_stations:
                final_stations.add(node)
                if len(final_stations) >= p:
                    break
    while len(final_stations) > p:
        # Remove station with least impact
        worst_node = None
        worst_loss = float('inf')
        for node in list(final_stations):
            test = final_stations - {node}
            test_obj, _ = compute_covered_flow(test)
            loss = best_objective - test_obj
            if loss < worst_loss:
                worst_loss = loss
                worst_node = node
        if worst_node is not None:
            final_stations.discard(worst_node)

    final_obj, final_covered = compute_covered_flow(final_stations)

    result = {
        "objective_value": final_obj,
        "stations": sorted(final_stations),
        "refueled_od_pairs": sorted(final_covered)
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