import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def find_cost(d):
    for k, v in d.items():
        if isinstance(v, (int, float)):
            k_lower = k.lower()
            if ("deploy" in k_lower or "vehicle" in k_lower or "fleet" in k_lower) and "cost" in k_lower:
                return v
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    topology = data["topology"]
    depot_id = topology["depot"]
    time_windows_data = {tw["index"]: tw for tw in data["time_windows"]}
    
    def get_pull_times(term, tw_idx):
        if term in data.get("pull_in_out_times", {}):
            for entry in data["pull_in_out_times"][term]:
                if entry["time_window_index"] == tw_idx:
                    return entry.get("pull_out_time_minutes", 0), entry.get("pull_in_time_minutes", 0)
        return 0, 0
        
    def get_stopping_time(term, tw_idx):
        if term in data.get("stopping_times", {}):
            for entry in data["stopping_times"][term]:
                if entry["time_window_index"] == tw_idx:
                    return entry.get("min_stopping_time_minutes", 0), entry.get("max_stopping_time_minutes", 999999)
        return 0, 999999

    obj_func = data["objective_function"]
    alpha = obj_func["alpha"]
    
    deploy_cost = data.get("per_vehicle_deployment_cost")
    if deploy_cost is None:
        deploy_cost = obj_func.get("per_vehicle_deployment_cost")
    if deploy_cost is None:
        deploy_cost = find_cost(data)
    if deploy_cost is None:
        deploy_cost = find_cost(obj_func)
    if deploy_cost is None:
        deploy_cost = 10000 

    desc = data.get("penalty_function", {}).get("description", "").lower()
    use_seconds_for_penalty = ("second" in desc)

    initial_tw = data["initial_trip_time_windows"]
    final_tw = data["final_trip_time_windows"]
    potential_trips = data["potential_trips"]
    patterns = list(set([t["pattern_id"] for t in potential_trips]))

    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    if args.time_limit:
        model.setParam("TimeLimit", args.time_limit)

    x_vars = {}
    for t in potential_trips:
        x_vars[t["id"]] = model.addVar(vtype=GRB.BINARY, name=f"x_{t['id']}")

    y_vars = {}
    tt_in = {t["id"]: [] for t in potential_trips}
    tt_out = {t["id"]: [] for t in potential_trips}
    tt_S_out = {pat: [] for pat in patterns}
    tt_E_in = {pat: [] for pat in patterns}

    for pat in patterns:
        pat_trips = [t for t in potential_trips if t["pattern_id"] == pat]
        pat_trips.sort(key=lambda t: t["main_stop_arrival_time_minutes"])
        
        for i in range(len(pat_trips)):
            t1 = pat_trips[i]
            tw1_idx = t1["time_window_index"]
            tw1 = time_windows_data[tw1_idx]
            
            if tw1_idx == initial_tw[pat]:
                var = model.addVar(vtype=GRB.BINARY, name=f"y_{pat}_S_{t1['id']}")
                y_vars[(pat, "source", t1["id"])] = var
                tt_S_out[pat].append(var)
                tt_in[t1["id"]].append(var)
                
            if tw1_idx == final_tw[pat]:
                var = model.addVar(vtype=GRB.BINARY, name=f"y_{pat}_{t1['id']}_E")
                y_vars[(pat, t1["id"], "sink")] = var
                tt_E_in[pat].append(var)
                tt_out[t1["id"]].append(var)
                
            for j in range(i + 1, len(pat_trips)):
                t2 = pat_trips[j]
                headway_mins = t2["main_stop_arrival_time_minutes"] - t1["main_stop_arrival_time_minutes"]
                headway_secs = headway_mins * 60
                
                if tw1["min_headway_seconds"] <= headway_secs <= tw1["max_headway_seconds"]:
                    if use_seconds_for_penalty:
                        dev = headway_secs - tw1["ideal_headway_seconds"]
                    else:
                        dev = headway_mins - (tw1["ideal_headway_seconds"] / 60.0)
                    cost = dev ** 2
                    
                    var = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"y_{pat}_{t1['id']}_{t2['id']}")
                    y_vars[(pat, t1["id"], t2["id"])] = var
                    tt_out[t1["id"]].append(var)
                    tt_in[t2["id"]].append(var)

    z_vars = {}
    vs_in = {t["id"]: [] for t in potential_trips}
    vs_out = {t["id"]: [] for t in potential_trips}
    
    V_var = model.addVar(vtype=GRB.INTEGER, obj=alpha * deploy_cost, name="num_vehicles")
    
    for t in potential_trips:
        pull_out, _ = get_pull_times(t["start_terminal"], t["time_window_index"])
        var = model.addVar(vtype=GRB.BINARY, obj=alpha * pull_out, name=f"z_Ominus_{t['id']}")
        u_str = "VS_O_minus"
        v_str = f"({t['id']}, 'start')"
        z_vars[(u_str, v_str)] = var
        vs_in[t["id"]].append(var)
        
        _, pull_in = get_pull_times(t["end_terminal"], t["time_window_index"])
        var = model.addVar(vtype=GRB.BINARY, obj=alpha * pull_in, name=f"z_{t['id']}_Oplus")
        u_str = f"({t['id']}, 'end')"
        v_str = "VS_O_plus"
        z_vars[(u_str, v_str)] = var
        vs_out[t["id"]].append(var)

    for t1 in potential_trips:
        for t2 in potential_trips:
            gap = t2["departure_time_minutes"] - t1["arrival_time_minutes"]
            if gap < 0:
                continue
                
            tw1 = t1["time_window_index"]
            tw2 = t2["time_window_index"]
            term1 = t1["end_terminal"]
            term2 = t2["start_terminal"]
            
            if term1 == term2:
                min_stop, max_stop = get_stopping_time(term1, tw1)
                if min_stop <= gap <= max_stop:
                    cost = alpha * (gap - min_stop)
                    var = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"z_{t1['id']}_{t2['id']}")
                    u_str = f"({t1['id']}, 'end')"
                    v_str = f"({t2['id']}, 'start')"
                    z_vars[(u_str, v_str)] = var
                    vs_out[t1["id"]].append(var)
                    vs_in[t2["id"]].append(var)
            else:
                _, pull_in_1 = get_pull_times(term1, tw1)
                pull_out_2, _ = get_pull_times(term2, tw2)
                min_depot, _ = get_stopping_time(depot_id, tw1)
                
                if gap >= pull_in_1 + min_depot + pull_out_2:
                    cost = alpha * (pull_in_1 + pull_out_2)
                    var = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"z_{t1['id']}_{t2['id']}")
                    u_str = f"({t1['id']}, 'end')"
                    v_str = f"({t2['id']}, 'start')"
                    z_vars[(u_str, v_str)] = var
                    vs_out[t1["id"]].append(var)
                    vs_in[t2["id"]].append(var)

    for pat in patterns:
        model.addConstr(gp.quicksum(tt_S_out[pat]) == 1, name=f"tt_start_{pat}")
        model.addConstr(gp.quicksum(tt_E_in[pat]) == 1, name=f"tt_end_{pat}")
        
    for t in potential_trips:
        tid = t["id"]
        model.addConstr(gp.quicksum(tt_in[tid]) == x_vars[tid], name=f"tt_in_{tid}")
        model.addConstr(gp.quicksum(tt_out[tid]) == x_vars[tid], name=f"tt_out_{tid}")
        model.addConstr(gp.quicksum(vs_in[tid]) == x_vars[tid], name=f"vs_in_{tid}")
        model.addConstr(gp.quicksum(vs_out[tid]) == x_vars[tid], name=f"vs_out_{tid}")

    model.addConstr(gp.quicksum(var for (u,v), var in z_vars.items() if u == "VS_O_minus") == V_var, name="depot_out")
    model.addConstr(gp.quicksum(var for (u,v), var in z_vars.items() if v == "VS_O_plus") == V_var, name="depot_in")

    def build_solution(model, is_mipsol=True):
        if is_mipsol:
            get_val = model.cbGetSolution
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        else:
            get_val = lambda v: v.X
            obj_val = model.ObjVal
            
        selected_trips = []
        for tid, var in x_vars.items():
            if get_val(var) > 0.5:
                selected_trips.append(tid)
                
        num_vehicles = int(round(get_val(V_var)))
        
        tt_arcs = {pat: [] for pat in patterns}
        for (pat, u, v), var in y_vars.items():
            if get_val(var) > 0.5:
                tt_arcs[pat].append([u, v])
                
        tt_arcs_used = {}
        for pat, arcs in tt_arcs.items():
            adj = {u: v for u, v in arcs}
            chain = []
            curr = "source"
            while curr in adj:
                nxt = adj[curr]
                chain.append([curr, nxt])
                curr = nxt
            tt_arcs_used[pat] = chain
            
        vs_flows = {}
        for (u, v), var in z_vars.items():
            val = get_val(var)
            if val > 0.5:
                vs_flows[f"{u}-->{v}"] = int(round(val))
                
        vs_flows["VS_O_plus-->VS_O_minus"] = num_vehicles
        for tid in selected_trips:
            vs_flows[f"({tid}, 'start')-->({tid}, 'end')"] = 1
            
        return obj_val, {
            "objective_value": obj_val,
            "selected_trips": selected_trips,
            "num_vehicles": num_vehicles,
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": vs_flows
        }

    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj, sol = build_solution(model, is_mipsol=True)
            if logger:
                logger.log_solution(obj, sol)

    model.optimize(my_callback)
    
    if model.SolCount > 0:
        obj, sol = build_solution(model, is_mipsol=False)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()