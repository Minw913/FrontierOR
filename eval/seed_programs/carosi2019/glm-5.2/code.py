import json
import argparse
import re
import time
import bisect
import gurobipy as gp
from gurobipy import GRB


def find_deployment_cost(obj, depth=0):
    """Search the instance recursively for a deployment cost field."""
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for key, val in obj.items():
            kl = key.lower()
            if any(w in kl for w in ['deployment', 'vehicle_cost', 'fleet_cost', 'per_vehicle']):
                if isinstance(val, (int, float)):
                    return float(val)
            result = find_deployment_cost(val, depth + 1)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_deployment_cost(item, depth + 1)
            if result is not None:
                return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    # Parse instance
    topology = instance['topology']
    depot = topology['depot']

    time_windows = instance['time_windows']
    tw_by_idx = {tw['index']: tw for tw in time_windows}

    stopping_times = instance.get('stopping_times', {})
    pull_in_out = instance.get('pull_in_out_times', {})

    initial_tw = instance.get('initial_trip_time_windows', {})
    final_tw = instance.get('final_trip_time_windows', {})

    obj_func = instance['objective_function']
    alpha = obj_func['alpha']

    deployment_cost = find_deployment_cost(instance)
    if deployment_cost is None:
        deployment_cost = 100000.0
    else:
        deployment_cost = float(deployment_cost)

    trips = instance['potential_trips']
    trips_by_pattern = {}
    for t in trips:
        pid = t['pattern_id']
        if pid not in trips_by_pattern:
            trips_by_pattern[pid] = []
        trips_by_pattern[pid].append(t)

    # Helper functions with caching
    stop_cache = {}

    def get_stopping_time(terminal, tw_idx):
        key = (terminal, tw_idx)
        if key in stop_cache:
            return stop_cache[key]
        if terminal in stopping_times:
            for st in stopping_times[terminal]:
                if st['time_window_index'] == tw_idx:
                    result = (st['min_stopping_time_minutes'], st['max_stopping_time_minutes'])
                    stop_cache[key] = result
                    return result
        result = (0, 999999)
        stop_cache[key] = result
        return result

    depot_stop_cache = {}

    def get_depot_min_stop(tw_idx):
        if tw_idx in depot_stop_cache:
            return depot_stop_cache[tw_idx]
        if depot in stopping_times:
            for st in stopping_times[depot]:
                if st['time_window_index'] == tw_idx:
                    depot_stop_cache[tw_idx] = st['min_stopping_time_minutes']
                    return st['min_stopping_time_minutes']
        depot_stop_cache[tw_idx] = 0
        return 0

    pull_cache = {}

    def get_pull_out(terminal, tw_idx):
        key = ('out', terminal, tw_idx)
        if key in pull_cache:
            return pull_cache[key]
        if terminal in pull_in_out:
            for pt in pull_in_out[terminal]:
                if pt['time_window_index'] == tw_idx:
                    pull_cache[key] = pt['pull_out_time_minutes']
                    return pt['pull_out_time_minutes']
        pull_cache[key] = 0
        return 0

    def get_pull_in(terminal, tw_idx):
        key = ('in', terminal, tw_idx)
        if key in pull_cache:
            return pull_cache[key]
        if terminal in pull_in_out:
            for pt in pull_in_out[terminal]:
                if pt['time_window_index'] == tw_idx:
                    pull_cache[key] = pt['pull_in_time_minutes']
                    return pt['pull_in_time_minutes']
        pull_cache[key] = 0
        return 0

    # Build timetable arcs for each pattern
    tt_arcs = {}
    for pid, pattern_trips in trips_by_pattern.items():
        arcs = []
        sorted_trips = sorted(pattern_trips, key=lambda t: t['main_stop_arrival_time_minutes'])
        init_tw_idx = initial_tw.get(pid)
        fin_tw_idx = final_tw.get(pid)

        for i, ti in enumerate(sorted_trips):
            tw_i = ti['time_window_index']
            tw_data = tw_by_idx[tw_i]
            min_h = tw_data['min_headway_seconds']
            max_h = tw_data['max_headway_seconds']
            ideal_h = tw_data['ideal_headway_seconds']

            if tw_i == init_tw_idx:
                arcs.append(('source', ti['id'], 0.0))
            if tw_i == fin_tw_idx:
                arcs.append((ti['id'], 'sink', 0.0))

            for j in range(i + 1, len(sorted_trips)):
                tj = sorted_trips[j]
                h = (tj['main_stop_arrival_time_minutes'] - ti['main_stop_arrival_time_minutes']) * 60
                if h < min_h:
                    continue
                if h > max_h:
                    break
                arcs.append((ti['id'], tj['id'], float((h - ideal_h) ** 2)))

        tt_arcs[pid] = arcs

    # Build VS arcs
    all_trips = sorted(trips, key=lambda t: t['arrival_time_minutes'])
    dep_sorted = sorted(trips, key=lambda t: t['departure_time_minutes'])
    dep_times = [t['departure_time_minutes'] for t in dep_sorted]

    vs_arc_list = []

    # Operation arcs
    for t in all_trips:
        tid = t['id']
        vs_arc_list.append((f"({tid}, 'start')", f"({tid}, 'end')", 0.0))

    # Pull-out and pull-in arcs
    for t in all_trips:
        tid = t['id']
        tw_i = t['time_window_index']
        start_node = f"({tid}, 'start')"
        end_node = f"({tid}, 'end')"

        po = float(get_pull_out(t['start_terminal'], tw_i))
        vs_arc_list.append(('VS_O_minus', start_node, po))

        pi = float(get_pull_in(t['end_terminal'], tw_i))
        vs_arc_list.append((end_node, 'VS_O_plus', pi))

    # Transition arcs (in-line and out-line)
    for ti in all_trips:
        end_term_i = ti['end_terminal']
        tw_i = ti['time_window_index']
        arr_i = ti['arrival_time_minutes']
        end_node_i = f"({ti['id']}, 'end')"

        min_stop, max_stop = get_stopping_time(end_term_i, tw_i)
        pull_in_i = float(get_pull_in(end_term_i, tw_i))
        depot_stop = get_depot_min_stop(tw_i)

        start_idx = bisect.bisect_left(dep_times, arr_i)

        for idx in range(start_idx, len(dep_sorted)):
            tj = dep_sorted[idx]
            if ti['id'] == tj['id']:
                continue
            start_term_j = tj['start_terminal']
            tw_j = tj['time_window_index']
            dep_j = tj['departure_time_minutes']
            gap = dep_j - arr_i

            start_node_j = f"({tj['id']}, 'start')"

            if end_term_i == start_term_j:
                # In-line compatible
                if min_stop <= gap <= max_stop:
                    vs_arc_list.append((end_node_i, start_node_j, float(gap - min_stop)))
            else:
                # Out-line compatible
                pull_out_j = float(get_pull_out(start_term_j, tw_j))
                min_required = pull_in_i + depot_stop + pull_out_j
                if gap >= min_required:
                    vs_arc_list.append((end_node_i, start_node_j, pull_in_i + pull_out_j))

    # Circulation arc
    vs_arc_list.append(('VS_O_plus', 'VS_O_minus', deployment_cost))

    # Precompute VS adjacency (excluding operation arcs)
    op_arc_set = set()
    for t in all_trips:
        tid = t['id']
        op_arc_set.add((f"({tid}, 'start')", f"({tid}, 'end')"))

    vs_incoming = {}
    vs_outgoing = {}
    for (a, b, cost) in vs_arc_list:
        key = (a, b)
        if key in op_arc_set:
            continue
        if b not in vs_incoming:
            vs_incoming[b] = []
        vs_incoming[b].append(key)
        if a not in vs_outgoing:
            vs_outgoing[a] = []
        vs_outgoing[a].append(key)

    # Build MIP
    build_time = time.time() - start_time
    remaining_time = max(1, args.time_limit - build_time - 3)

    model = gp.Model('integrated')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('OutputFlag', 0)

    # Variables
    z = {t['id']: model.addVar(vtype=GRB.BINARY, name=f'z_{t["id"]}') for t in trips}

    y = {}
    for pid, arcs in tt_arcs.items():
        for (a, b, cost) in arcs:
            y[(pid, a, b)] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f'y_{pid}_{a}_{b}')

    f = {}
    for (a, b, cost) in vs_arc_list:
        if (a, b) not in f:
            f[(a, b)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=alpha * cost, name=f'f_{a}_{b}')

    model.update()

    # Timetable constraints
    for pid, arcs in tt_arcs.items():
        pattern_trip_ids = set(t['id'] for t in trips_by_pattern[pid])

        source_out = [y[(pid, a, b)] for (a, b, cost) in arcs if a == 'source']
        if source_out:
            model.addConstr(gp.quicksum(source_out) == 1, name=f'tt_src_{pid}')

        sink_in = [y[(pid, a, b)] for (a, b, cost) in arcs if b == 'sink']
        if sink_in:
            model.addConstr(gp.quicksum(sink_in) == 1, name=f'tt_snk_{pid}')

        for tid in pattern_trip_ids:
            incoming = [y[(pid, a, b)] for (a, b, cost) in arcs if b == tid]
            outgoing = [y[(pid, a, b)] for (a, b, cost) in arcs if a == tid]

            if incoming:
                model.addConstr(z[tid] == gp.quicksum(incoming), name=f'tt_z_{pid}_{tid}')
            else:
                model.addConstr(z[tid] == 0, name=f'tt_z_{pid}_{tid}')

            if incoming and outgoing:
                model.addConstr(gp.quicksum(incoming) == gp.quicksum(outgoing), name=f'tt_fc_{pid}_{tid}')
            elif outgoing and not incoming:
                model.addConstr(gp.quicksum(outgoing) == 0, name=f'tt_fc_{pid}_{tid}')

    # VS constraints
    for t in trips:
        tid = t['id']
        start_node = f"({tid}, 'start')"
        end_node = f"({tid}, 'end')"

        op_key = (start_node, end_node)
        if op_key in f:
            model.addConstr(f[op_key] == z[tid], name=f'vs_op_{tid}')

        in_start = [f[key] for key in vs_incoming.get(start_node, [])]
        if in_start:
            model.addConstr(gp.quicksum(in_start) == z[tid], name=f'vs_in_{tid}')

        out_end = [f[key] for key in vs_outgoing.get(end_node, [])]
        if out_end:
            model.addConstr(gp.quicksum(out_end) == z[tid], name=f'vs_out_{tid}')

    # Depot flow conservation
    d_minus_out = [f[key] for key in vs_outgoing.get('VS_O_minus', [])]
    d_minus_in = [f[key] for key in vs_incoming.get('VS_O_minus', [])]
    d_plus_in = [f[key] for key in vs_incoming.get('VS_O_plus', [])]
    d_plus_out = [f[key] for key in vs_outgoing.get('VS_O_plus', [])]

    if d_minus_out or d_minus_in:
        model.addConstr(gp.quicksum(d_minus_out) == gp.quicksum(d_minus_in), name='depot_minus')
    if d_plus_in or d_plus_out:
        model.addConstr(gp.quicksum(d_plus_in) == gp.quicksum(d_plus_out), name='depot_plus')

    model.modelSense = GRB.MINIMIZE

    # Precompute TT adjacency for chain tracing
    tt_adj = {}
    for pid, arcs in tt_arcs.items():
        tt_adj[pid] = {}
        for (a, b, cost) in arcs:
            if a not in tt_adj[pid]:
                tt_adj[pid][a] = []
            tt_adj[pid][a].append((b, (pid, a, b)))

    # Solution tracking
    best_obj = [float('inf')]
    num_vars = len(z) + len(y) + len(f)
    build_full_callback = num_vars < 300000

    def build_sol_dict(z_vals, y_vals, f_vals, obj_val):
        selected = sorted([tid for tid, v in z_vals.items() if v > 0.5])

        tt_arcs_used = {}
        for pid in tt_arcs:
            chain = []
            current = 'source'
            visited = set()
            while current != 'sink' and current in tt_adj.get(pid, {}):
                found = False
                for (b, ykey) in tt_adj[pid][current]:
                    if ykey in y_vals and y_vals[ykey] > 0.5:
                        chain.append([current, b])
                        current = b
                        found = True
                        break
                if not found:
                    break
                if current in visited:
                    break
                visited.add(current)
            tt_arcs_used[pid] = chain

        vs_flows = {}
        num_veh = 0
        for key, val in f_vals.items():
            if val > 1e-6:
                a, b = key
                vs_flows[f"{a}-->{b}"] = val
                if a == 'VS_O_plus' and b == 'VS_O_minus':
                    num_veh = int(round(val))

        return {
            'objective_value': obj_val,
            'selected_trips': selected,
            'num_vehicles': num_veh,
            'tt_arcs_used': tt_arcs_used,
            'vs_flows': vs_flows
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                if logger:
                    if build_full_callback:
                        try:
                            z_vals = {tid: model.cbGetSolution(var) for tid, var in z.items()}
                            y_vals = {key: model.cbGetSolution(var) for key, var in y.items()}
                            f_vals = {key: model.cbGetSolution(var) for key, var in f.items()}
                            sol_dict = build_sol_dict(z_vals, y_vals, f_vals, obj)
                            logger.log_solution(obj, sol_dict)
                        except Exception:
                            logger.log(obj)
                    else:
                        logger.log(obj)

    model.optimize(callback)

    # Extract final solution
    sol_dict = None
    if model.SolCount > 0:
        obj_val = model.ObjVal
        z_vals = {tid: var.X for tid, var in z.items()}
        y_vals = {key: var.X for key, var in y.items()}
        f_vals = {key: var.X for key, var in f.items()}
        sol_dict = build_sol_dict(z_vals, y_vals, f_vals, obj_val)

        if logger and obj_val < best_obj[0]:
            logger.log_solution(obj_val, sol_dict)

    if sol_dict is None:
        sol_dict = {
            'objective_value': 0.0,
            'selected_trips': [],
            'num_vehicles': 0,
            'tt_arcs_used': {pid: [] for pid in tt_arcs},
            'vs_flows': {}
        }

    with open(args.solution_path, 'w') as fout:
        json.dump(sol_dict, fout, indent=2)


if __name__ == '__main__':
    main()