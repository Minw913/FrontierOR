import argparse
import bisect
import json
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_nested_numeric(instance: Dict[str, Any], candidate_keys: List[str], default: float) -> float:
    # First check objective_function
    objf = instance.get("objective_function", {})
    for k in candidate_keys:
        if k in objf and isinstance(objf[k], (int, float)):
            return float(objf[k])

    # Then top-level
    for k in candidate_keys:
        if k in instance and isinstance(instance[k], (int, float)):
            return float(instance[k])

    return float(default)


def build_solution(
    pattern_ids,
    tt_vars_by_pattern,
    tt_source_node,
    tt_sink_node,
    y_vars,
    vs_start_vars,
    vs_end_vars,
    vs_arc_vars,
    n_var,
    get_val,
    obj_val,
):
    selected_trips = sorted([tid for tid, var in y_vars.items() if get_val(var) > 0.5])

    # Timetable arcs used
    tt_arcs_used = {}
    for p in pattern_ids:
        succ = {}
        selected_arcs = []
        for (u, v), var in tt_vars_by_pattern[p].items():
            if get_val(var) > 0.5:
                succ[u] = v
                selected_arcs.append((u, v))

        source = tt_source_node[p]
        sink = tt_sink_node[p]
        ordered = []

        if source in succ:
            u = source
            max_steps = len(selected_arcs) + 5
            steps = 0
            while u in succ and steps < max_steps:
                v = succ[u]
                ordered.append([str(u), str(v)])
                u = v
                steps += 1
                if u == sink:
                    break
        else:
            ordered = [[str(u), str(v)] for (u, v) in selected_arcs]

        tt_arcs_used[p] = ordered

    # VS flows
    vs_flows = {}

    num_vehicles = int(round(get_val(n_var)))
    vs_flows["VS_O_plus-->VS_O_minus"] = num_vehicles

    # Operation arcs for selected trips
    for tid in selected_trips:
        flow = get_val(y_vars[tid])
        if flow > 1e-6:
            key = f"{str((tid, 'start'))}-->{str((tid, 'end'))}"
            vs_flows[key] = float(round(flow))

    # Depot pull-out arcs
    for tid, var in vs_start_vars.items():
        flow = get_val(var)
        if flow > 1e-6:
            key = f"VS_O_minus-->{str((tid, 'start'))}"
            vs_flows[key] = float(round(flow))

    # Depot pull-in arcs
    for tid, var in vs_end_vars.items():
        flow = get_val(var)
        if flow > 1e-6:
            key = f"{str((tid, 'end'))}-->VS_O_plus"
            vs_flows[key] = float(round(flow))

    # Inter-trip compatibility arcs
    for (i, j), var in vs_arc_vars.items():
        flow = get_val(var)
        if flow > 1e-6:
            key = f"{str((i, 'end'))}-->{str((j, 'start'))}"
            vs_flows[key] = float(round(flow))

    sol = {
        "objective_value": float(obj_val),
        "selected_trips": selected_trips,
        "num_vehicles": num_vehicles,
        "tt_arcs_used": tt_arcs_used,
        "vs_flows": vs_flows,
    }
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    directions = instance.get("directions", [])
    potential_trips = instance.get("potential_trips", [])
    time_windows = instance.get("time_windows", [])
    stopping_times_raw = instance.get("stopping_times", {})
    pull_raw = instance.get("pull_in_out_times", {})
    initial_tw_map = instance.get("initial_trip_time_windows", {})
    final_tw_map = instance.get("final_trip_time_windows", {})
    depot = instance.get("topology", {}).get("depot", "O")

    alpha = float(instance.get("objective_function", {}).get("alpha", 1.0))
    deploy_cost = get_nested_numeric(
        instance,
        [
            "vehicle_deployment_cost",
            "per_vehicle_deployment_cost",
            "fixed_vehicle_cost",
            "cost_per_vehicle",
            "vehicle_cost",
        ],
        default=10000.0,
    )

    # Time window maps
    tw_min_headway = {}
    tw_max_headway = {}
    tw_ideal_headway = {}
    for tw in time_windows:
        idx = tw["index"]
        tw_min_headway[idx] = float(tw["min_headway_seconds"])
        tw_max_headway[idx] = float(tw["max_headway_seconds"])
        tw_ideal_headway[idx] = float(tw["ideal_headway_seconds"])

    # Stopping times by terminal, tw
    stop_min = defaultdict(lambda: defaultdict(lambda: 0.0))
    stop_max = defaultdict(lambda: defaultdict(lambda: 1e9))
    for term, arr in stopping_times_raw.items():
        for rec in arr:
            w = rec["time_window_index"]
            stop_min[term][w] = float(rec["min_stopping_time_minutes"])
            stop_max[term][w] = float(rec["max_stopping_time_minutes"])

    # Depot minimum stop by tw (if provided), else 0
    depot_min_stop = defaultdict(lambda: 0.0)
    if depot in stopping_times_raw:
        for rec in stopping_times_raw[depot]:
            depot_min_stop[rec["time_window_index"]] = float(rec["min_stopping_time_minutes"])

    # Pull-in/out maps
    pull_out = defaultdict(lambda: defaultdict(lambda: 0.0))
    pull_in = defaultdict(lambda: defaultdict(lambda: 0.0))
    for term, arr in pull_raw.items():
        for rec in arr:
            w = rec["time_window_index"]
            pull_out[term][w] = float(rec["pull_out_time_minutes"])
            pull_in[term][w] = float(rec["pull_in_time_minutes"])

    # Trips by id and by pattern
    trip_by_id = {}
    trips_by_pattern = defaultdict(list)
    for t in potential_trips:
        tid = int(t["id"])
        trip_by_id[tid] = t
        trips_by_pattern[t["pattern_id"]].append(tid)

    pattern_ids = []
    for d in directions:
        pid = d["pattern_id"]
        if pid not in pattern_ids:
            pattern_ids.append(pid)

    # Build model
    m = gp.Model("integrated_timetable_vehicle_schedule")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = float(args.time_limit)

    # Variables
    y_vars = {tid: m.addVar(vtype=GRB.BINARY, name=f"y_{tid}") for tid in trip_by_id.keys()}

    # Timetable variables
    tt_vars_by_pattern = {p: {} for p in pattern_ids}
    tt_source_node = {p: f"TT_{p}_source" for p in pattern_ids}
    tt_sink_node = {p: f"TT_{p}_sink" for p in pattern_ids}
    tt_penalty_terms = []

    for p in pattern_ids:
        tids = trips_by_pattern.get(p, [])
        source = tt_source_node[p]
        sink = tt_sink_node[p]

        # Build sorted list by main-stop time
        tids_sorted = sorted(tids, key=lambda tid: (trip_by_id[tid]["main_stop_arrival_time_minutes"], tid))
        in_arcs = defaultdict(list)
        out_arcs = defaultdict(list)

        init_tw = initial_tw_map.get(p, None)
        final_tw = final_tw_map.get(p, None)

        init_candidates = []
        final_candidates = []
        for tid in tids_sorted:
            tw = trip_by_id[tid]["time_window_index"]
            if init_tw is not None and tw == init_tw:
                init_candidates.append(tid)
            if final_tw is not None and tw == final_tw:
                final_candidates.append(tid)

        # source -> initial candidates
        for j in init_candidates:
            var = m.addVar(vtype=GRB.BINARY, name=f"x_{p}_{source}_{j}")
            tt_vars_by_pattern[p][(source, j)] = var
            in_arcs[j].append(var)

        # final candidates -> sink
        for i in final_candidates:
            var = m.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}_{sink}")
            tt_vars_by_pattern[p][(i, sink)] = var
            out_arcs[i].append(var)

        # trip -> trip arcs with headway feasibility
        n = len(tids_sorted)
        for a in range(n):
            i = tids_sorted[a]
            ti = trip_by_id[i]
            tw_i = ti["time_window_index"]
            t_i_main = float(ti["main_stop_arrival_time_minutes"])
            hmin = tw_min_headway[tw_i]
            hmax = tw_max_headway[tw_i]
            ideal = tw_ideal_headway[tw_i]

            for b in range(a + 1, n):
                j = tids_sorted[b]
                tj = trip_by_id[j]
                t_j_main = float(tj["main_stop_arrival_time_minutes"])
                hw_sec = (t_j_main - t_i_main) * 60.0

                if hw_sec < hmin:
                    continue
                if hw_sec > hmax:
                    break  # sorted in time

                var = m.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}_{j}")
                tt_vars_by_pattern[p][(i, j)] = var
                out_arcs[i].append(var)
                in_arcs[j].append(var)

                penalty = (hw_sec - ideal) ** 2
                tt_penalty_terms.append(penalty * var)

        # chain constraints
        source_out = gp.quicksum(var for (u, v), var in tt_vars_by_pattern[p].items() if u == source)
        sink_in = gp.quicksum(var for (u, v), var in tt_vars_by_pattern[p].items() if v == sink)
        m.addConstr(source_out == 1, name=f"tt_source_out_{p}")
        m.addConstr(sink_in == 1, name=f"tt_sink_in_{p}")

        for tid in tids_sorted:
            m.addConstr(gp.quicksum(in_arcs[tid]) == y_vars[tid], name=f"tt_in_eq_y_{p}_{tid}")
            m.addConstr(gp.quicksum(out_arcs[tid]) == y_vars[tid], name=f"tt_out_eq_y_{p}_{tid}")

    # Vehicle scheduling variables
    vs_start_vars = {tid: m.addVar(vtype=GRB.BINARY, name=f"vs_start_{tid}") for tid in trip_by_id.keys()}
    vs_end_vars = {tid: m.addVar(vtype=GRB.BINARY, name=f"vs_end_{tid}") for tid in trip_by_id.keys()}
    n_var = m.addVar(vtype=GRB.INTEGER, lb=0, name="num_vehicles")

    vs_arc_vars = {}
    vs_arc_cost = {}

    # Prepare sorted-by-departure for arc generation
    tids_all = list(trip_by_id.keys())
    dep_sorted = sorted(tids_all, key=lambda tid: (trip_by_id[tid]["departure_time_minutes"], tid))
    dep_times = [trip_by_id[tid]["departure_time_minutes"] for tid in dep_sorted]

    incoming_vs = defaultdict(list)
    outgoing_vs = defaultdict(list)

    for i in tids_all:
        ti = trip_by_id[i]
        arr_i = float(ti["arrival_time_minutes"])
        tw_i = ti["time_window_index"]
        end_i = ti["end_terminal"]

        start_idx = bisect.bisect_left(dep_times, arr_i)
        for pos in range(start_idx, len(dep_sorted)):
            j = dep_sorted[pos]
            if i == j:
                continue
            tj = trip_by_id[j]
            dep_j = float(tj["departure_time_minutes"])
            gap = dep_j - arr_i
            if gap < 0:
                continue

            start_j = tj["start_terminal"]
            tw_j = tj["time_window_index"]

            feasible = False
            cost = 0.0

            if end_i == start_j:
                mn = stop_min[end_i][tw_i]
                mx = stop_max[end_i][tw_i]
                if gap >= mn - 1e-9 and gap <= mx + 1e-9:
                    feasible = True
                    cost = gap - mn
            else:
                req = pull_in[end_i][tw_i] + depot_min_stop[tw_i] + pull_out[start_j][tw_j]
                if gap >= req - 1e-9:
                    feasible = True
                    cost = pull_in[end_i][tw_i] + pull_out[start_j][tw_j]

            if feasible:
                var = m.addVar(vtype=GRB.BINARY, name=f"vs_{i}_{j}")
                vs_arc_vars[(i, j)] = var
                vs_arc_cost[(i, j)] = cost
                outgoing_vs[i].append(var)
                incoming_vs[j].append(var)

    # Coverage/flow constraints for vehicle paths
    for tid in tids_all:
        m.addConstr(vs_start_vars[tid] + gp.quicksum(incoming_vs[tid]) == y_vars[tid], name=f"vs_pred_{tid}")
        m.addConstr(vs_end_vars[tid] + gp.quicksum(outgoing_vs[tid]) == y_vars[tid], name=f"vs_succ_{tid}")

    m.addConstr(n_var == gp.quicksum(vs_start_vars[tid] for tid in tids_all), name="num_veh_def")

    # Objective
    vs_cost_expr = deploy_cost * n_var

    # start/end pull costs
    for tid in tids_all:
        t = trip_by_id[tid]
        tw = t["time_window_index"]
        sterm = t["start_terminal"]
        eterm = t["end_terminal"]
        vs_cost_expr += pull_out[sterm][tw] * vs_start_vars[tid]
        vs_cost_expr += pull_in[eterm][tw] * vs_end_vars[tid]

    # compatibility costs
    for (i, j), var in vs_arc_vars.items():
        vs_cost_expr += vs_arc_cost[(i, j)] * var

    tt_cost_expr = gp.quicksum(tt_penalty_terms)

    m.setObjective(alpha * vs_cost_expr + tt_cost_expr, GRB.MINIMIZE)

    # Prepare callback data
    m._logger = logger
    m._pattern_ids = pattern_ids
    m._tt_vars_by_pattern = tt_vars_by_pattern
    m._tt_source_node = tt_source_node
    m._tt_sink_node = tt_sink_node
    m._y_vars = y_vars
    m._vs_start_vars = vs_start_vars
    m._vs_end_vars = vs_end_vars
    m._vs_arc_vars = vs_arc_vars
    m._n_var = n_var

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

            def cb_get(v):
                return model.cbGetSolution(v)

            sol = build_solution(
                model._pattern_ids,
                model._tt_vars_by_pattern,
                model._tt_source_node,
                model._tt_sink_node,
                model._y_vars,
                model._vs_start_vars,
                model._vs_end_vars,
                model._vs_arc_vars,
                model._n_var,
                cb_get,
                obj,
            )
            model._logger.log_solution(float(obj), sol)

    if logger is not None:
        m.optimize(cb)
    else:
        m.optimize()

    # Final solution
    if m.SolCount > 0:
        final_obj = float(m.ObjVal)

        def get_x(v):
            return v.X

        solution = build_solution(
            pattern_ids,
            tt_vars_by_pattern,
            tt_source_node,
            tt_sink_node,
            y_vars,
            vs_start_vars,
            vs_end_vars,
            vs_arc_vars,
            n_var,
            get_x,
            final_obj,
        )
    else:
        solution = {
            "objective_value": 1e30,
            "selected_trips": [],
            "num_vehicles": 0,
            "tt_arcs_used": {p: [] for p in pattern_ids},
            "vs_flows": {"VS_O_plus-->VS_O_minus": 0},
        }
        if logger is not None:
            logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()