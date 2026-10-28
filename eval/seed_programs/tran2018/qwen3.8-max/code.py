import json
import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    params = instance['parameters']
    n = params['n']
    R = params['R']
    p = params['p']

    nodes = instance['nodes']
    arcs = instance['arcs']
    od_pairs = instance['od_pairs']

    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    # Build distance lookup
    arc_dist = {}
    for arc in arcs:
        u, v, d = arc['from'], arc['to'], arc['distance']
        arc_dist[(u, v)] = d
        arc_dist[(v, u)] = d

    all_node_ids = [node['id'] for node in nodes]

    # Preprocess OD pairs
    auto_refueled_flow = 0.0
    auto_refueled_pairs = []
    needs_station_info = []  # (q_idx, flow, list of serviceable node sets)
    all_candidate_nodes = set()

    for q_idx, od_pair in enumerate(od_pairs):
        path = od_pair['shortest_path']
        flow = od_pair['flow']

        if len(path) <= 1:
            auto_refueled_flow += flow
            auto_refueled_pairs.append(q_idx)
            continue

        # Round-trip path: forward then reverse (no duplicate at destination)
        round_trip = path + path[-2::-1]

        # Compute cumulative distances along round-trip path
        cum_dist = [0.0]
        for i in range(1, len(round_trip)):
            u, v = round_trip[i - 1], round_trip[i]
            d = arc_dist.get((u, v), 0.0)
            cum_dist.append(cum_dist[-1] + d)

        # Find critical arcs (those beyond initial range R) and their serviceable nodes
        critical_constraints = []
        feasible = True

        for i in range(1, len(round_trip)):
            if cum_dist[i] > R + 1e-9:
                serviceable = set()
                for j in range(i):
                    if cum_dist[i] - cum_dist[j] <= R + 1e-9:
                        serviceable.add(round_trip[j])
                if not serviceable:
                    feasible = False
                    break
                critical_constraints.append(serviceable)
                all_candidate_nodes.update(serviceable)

        if not feasible:
            continue  # This OD pair can never be fully refueled

        if not critical_constraints:
            auto_refueled_flow += flow
            auto_refueled_pairs.append(q_idx)
        else:
            needs_station_info.append((q_idx, flow, critical_constraints))

    def make_solution(stations, refueled_pairs, obj_val):
        return {
            "objective_value": obj_val,
            "stations": stations[:p],
            "refueled_od_pairs": sorted(refueled_pairs)
        }

    def pad_stations(stations_list):
        """Pad station list to exactly p nodes."""
        result = list(stations_list)
        for nid in all_node_ids:
            if len(result) >= p:
                break
            if nid not in result:
                result.append(nid)
        return result[:p]

    # Greedy heuristic for initial solution
    def greedy_solve(time_budget):
        greedy_start = time.time()
        open_stations = set()
        current_flow = auto_refueled_flow
        refueled = set(auto_refueled_pairs)

        # Precompute: for each OD pair needing stations, which candidate nodes appear in its constraints
        od_candidate_nodes = {}
        for (q_idx, flow, constraints) in needs_station_info:
            node_set = set()
            for s in constraints:
                node_set.update(s)
            od_candidate_nodes[q_idx] = node_set

        for _ in range(min(p, len(all_candidate_nodes))):
            if time.time() - greedy_start > time_budget:
                break

            best_node = None
            best_gain = -1.0

            for node in all_candidate_nodes:
                if node in open_stations:
                    continue

                gain = 0.0
                test_set = open_stations | {node}

                for (q_idx, flow, constraints) in needs_station_info:
                    if q_idx in refueled:
                        continue
                    # Quick check: does this node even appear in any constraint of this OD pair?
                    if node not in od_candidate_nodes[q_idx]:
                        continue
                    # Check if all critical arcs are covered
                    all_covered = True
                    for serviceable in constraints:
                        if not (serviceable & test_set):
                            all_covered = False
                            break
                    if all_covered:
                        gain += flow

                if gain > best_gain:
                    best_gain = gain
                    best_node = node

            if best_node is None or best_gain <= 0:
                break

            open_stations.add(best_node)
            current_flow += best_gain

            # Update refueled set
            for (q_idx, flow, constraints) in needs_station_info:
                if q_idx in refueled:
                    continue
                all_covered = True
                for serviceable in constraints:
                    if not (serviceable & open_stations):
                        all_covered = False
                        break
                if all_covered:
                    refueled.add(q_idx)

        return list(open_stations), list(refueled), current_flow

    # Run greedy heuristic with limited time
    greedy_time_budget = min(5.0, args.time_limit * 0.1)
    greedy_stations, greedy_refueled, greedy_flow = greedy_solve(greedy_time_budget)
    greedy_stations_padded = pad_stations(greedy_stations)

    best_solution = make_solution(greedy_stations_padded, greedy_refueled, greedy_flow)
    best_obj = greedy_flow

    if logger:
        logger.log_solution(greedy_flow, best_solution)

    # Solve MIP with Gurobi
    remaining_time = args.time_limit - (time.time() - start_time)

    if remaining_time > 2 and len(needs_station_info) > 0 and len(all_candidate_nodes) > 0:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("refueling")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(remaining_time - 1, 1))
        model.setParam('OutputFlag', 0)

        candidate_nodes = sorted(all_candidate_nodes)

        # Decision variables
        x = {}
        for node in candidate_nodes:
            x[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

        y = {}
        for (q_idx, flow, constraints) in needs_station_info:
            y[q_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{q_idx}")

        model.update()

        # Objective: maximize total refueled flow
        obj = gp.LinExpr(auto_refueled_flow)
        for (q_idx, flow, constraints) in needs_station_info:
            obj += flow * y[q_idx]
        model.setObjective(obj, GRB.MAXIMIZE)

        # Cardinality constraint
        model.addConstr(gp.quicksum(x[node] for node in candidate_nodes) <= p)

        # Coverage constraints: for each OD pair and each critical arc,
        # at least one serviceable station must be open
        for (q_idx, flow, constraints) in needs_station_info:
            for serviceable in constraints:
                model.addConstr(
                    y[q_idx] <= gp.quicksum(x[node] for node in serviceable)
                )

        # Warm start from greedy solution
        greedy_station_set = set(greedy_stations)
        greedy_refueled_set = set(greedy_refueled)
        for node in candidate_nodes:
            x[node].Start = 1.0 if node in greedy_station_set else 0.0
        for (q_idx, flow, constraints) in needs_station_info:
            y[q_idx].Start = 1.0 if q_idx in greedy_refueled_set else 0.0

        # Callback to log incumbent solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > best_obj + 1e-6:
                    sel_stations = [node for node in candidate_nodes
                                    if model.cbGetSolution(x[node]) > 0.5]
                    refueled = list(auto_refueled_pairs)
                    for (q_idx, flow, constraints) in needs_station_info:
                        if model.cbGetSolution(y[q_idx]) > 0.5:
                            refueled.append(q_idx)

                    sel_stations_padded = pad_stations(sel_stations)
                    sol = make_solution(sel_stations_padded, refueled, obj_val)

                    if logger:
                        logger.log_solution(obj_val, sol)

        model.optimize(callback)

        # Extract final solution from Gurobi
        if model.SolCount > 0:
            final_obj = model.ObjVal
            if final_obj > best_obj + 1e-6:
                sel_stations = [node for node in candidate_nodes if x[node].X > 0.5]
                refueled = list(auto_refueled_pairs)
                for (q_idx, flow, constraints) in needs_station_info:
                    if y[q_idx].X > 0.5:
                        refueled.append(q_idx)

                sel_stations_padded = pad_stations(sel_stations)
                best_obj = final_obj
                best_solution = make_solution(sel_stations_padded, refueled, final_obj)

                if logger:
                    logger.log_solution(final_obj, best_solution)
            elif abs(final_obj - best_obj) <= 1e-6:
                # Same objective, but let's still update if Gurobi found a solution
                sel_stations = [node for node in candidate_nodes if x[node].X > 0.5]
                refueled = list(auto_refueled_pairs)
                for (q_idx, flow, constraints) in needs_station_info:
                    if y[q_idx].X > 0.5:
                        refueled.append(q_idx)
                sel_stations_padded = pad_stations(sel_stations)
                best_solution = make_solution(sel_stations_padded, refueled, final_obj)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()