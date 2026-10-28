import argparse
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    start_time_total = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    depot_data = instance["depot"]
    veh_cap = instance["vehicle_capacity"]
    customers_data = instance["customers"]
    scenarios_data = instance["scenarios"]
    dist_matrix = instance["distance_matrix"]

    depot_exo = depot_data["exogenous_time_window"]
    
    clients = {}
    for c in customers_data:
        clients[c["id"]] = {
            "id": c["id"],
            "exo_start": c["exogenous_time_window"][0],
            "exo_end": c["exogenous_time_window"][1],
            "endo_width": c["endogenous_time_window_width"]
        }
        
    demands_max = {c["id"]: 0 for c in customers_data}
    for s in scenarios_data:
        dems = s["demands"]
        for j in range(num_customers):
            cid = customers_data[j]["id"]
            demands_max[cid] = max(demands_max[cid], dems[j])

    V_C = [c["id"] for c in customers_data]
    N = num_customers

    best_obj = float('inf')
    best_sol = None

    def log_solution(sol):
        nonlocal best_obj, best_sol
        if sol["objective_value"] < best_obj - 1e-4:
            best_obj = sol["objective_value"]
            best_sol = sol
            if logger:
                logger.log(best_obj)

    def build_greedy_universal(scenarios_list, demands_max_local):
        unvisited = set(V_C)
        routes = []
        
        while unvisited:
            route = []
            curr_loc = 0
            curr_time = depot_exo[0]
            curr_load = 0
            
            prev_unvisited = len(unvisited)
            while unvisited:
                best_j = None
                best_cost = float('inf')
                
                for j in unvisited:
                    if curr_load + demands_max_local[j] > veh_cap:
                        continue
                    
                    arr_time = curr_time + dist_matrix[curr_loc][j]
                    serv_time = max(arr_time, clients[j]['exo_start'])
                    
                    if serv_time > clients[j]['exo_end']:
                        continue
                    
                    if serv_time + dist_matrix[j][-1] > depot_exo[1]:
                        continue
                        
                    cost = dist_matrix[curr_loc][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
                        
                if best_j is not None:
                    route.append(best_j)
                    arr_time = curr_time + dist_matrix[curr_loc][best_j]
                    curr_time = max(arr_time, clients[best_j]['exo_start'])
                    curr_load += demands_max_local[best_j]
                    curr_loc = best_j
                    unvisited.remove(best_j)
                else:
                    break
            
            if len(unvisited) == prev_unvisited:
                return None  
            routes.append(route)
            
        t_i = {}
        obj_cost = 0
        route_details = []
        
        for route in routes:
            curr_loc = 0
            curr_time = depot_exo[0]
            route_cost = 0
            
            arr_times = {"0": curr_time}
            
            for j in route:
                arr_time = curr_time + dist_matrix[curr_loc][j]
                serv_time = max(arr_time, clients[j]['exo_start'])
                t_i[j] = serv_time
                route_cost += dist_matrix[curr_loc][j]
                arr_times[str(clients[j]['id'])] = serv_time
                curr_time = serv_time
                curr_loc = j
                
            arr_time = curr_time + dist_matrix[curr_loc][-1]
            route_cost += dist_matrix[curr_loc][-1]
            arr_times[str(N + 1)] = arr_time
            obj_cost += route_cost
            
            route_nodes = [0] + route + [N + 1]
            route_details.append({
                "route": route_nodes,
                "arrival_times": arr_times,
                "cost": route_cost
            })
            
        time_windows = {}
        for i in V_C:
            serv_time = t_i[i]
            W_i = clients[i]['endo_width']
            t_start = serv_time - W_i / 2.0
            t_start = max(t_start, clients[i]['exo_start'])
            t_start = min(t_start, clients[i]['exo_end'] - W_i)
            time_windows[str(clients[i]['id'])] = [t_start, t_start + W_i]
            
        routes_output = {}
        for s in scenarios_list:
            routes_output[str(s['scenario_id'])] = route_details
            
        return {
            "objective_value": obj_cost,
            "time_windows": time_windows,
            "routes": routes_output
        }

    greedy_sol = build_greedy_universal(scenarios_data, demands_max)
    if greedy_sol is not None:
        log_solution(greedy_sol)

    def map_single_to_multi(sol, all_scenarios):
        new_sol = {
            "objective_value": sol["objective_value"],
            "time_windows": sol["time_windows"],
            "routes": {}
        }
        single_route_list = list(sol["routes"].values())[0]
        for s in all_scenarios:
            new_sol["routes"][str(s['scenario_id'])] = single_route_list
        return new_sol

    def solve_routing_model(scenarios_list, time_limit_sec, mip_start_sol=None, is_multi=False):
        import gurobipy as gp
        from gurobipy import GRB
        
        S_local = len(scenarios_list)
        
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("Delivery_VRP", env=env)
        model.setParam('TimeLimit', max(1.0, time_limit_sec))
        model.setParam('Threads', 1)
        model.setParam('MIPGap', 1e-4)

        dems_local = {}
        probs_local = {}
        for s_idx, s_data in enumerate(scenarios_list):
            probs_local[s_idx] = s_data["probability"]
            dems_local[s_idx] = {}
            for j in range(num_customers):
                cid = customers_data[j]["id"]
                dems_local[s_idx][cid] = s_data["demands"][j]

        T_start = {}
        for i in V_C:
            e_start = clients[i]['exo_start']
            e_end = clients[i]['exo_end']
            w_i = clients[i]['endo_width']
            T_start[i] = model.addVar(lb=e_start, ub=e_end - w_i, vtype=GRB.CONTINUOUS, name=f"T_start_{i}")

        x = {}
        t = {}
        u = {}

        for s in range(S_local):
            for i in range(N + 2):
                for j in range(N + 2):
                    if i == j: continue
                    if i == 0 and j == N + 1: continue
                    if i == N + 1: continue
                    if j == 0: continue
                    x[i, j, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{s}")
                    
            for i in V_C:
                t[i, s] = model.addVar(lb=clients[i]['exo_start'], ub=clients[i]['exo_end'], vtype=GRB.CONTINUOUS, name=f"t_{i}_{s}")
                u[i, s] = model.addVar(lb=dems_local[s][i], ub=veh_cap, vtype=GRB.CONTINUOUS, name=f"u_{i}_{s}")

        obj_expr = gp.LinExpr()
        for s in range(S_local):
            for (i, j, _s), var in x.items():
                if _s == s:
                    obj_expr += probs_local[s] * dist_matrix[i][j] * var
        model.setObjective(obj_expr, GRB.MINIMIZE)

        for s in range(S_local):
            for j in V_C:
                model.addConstr(gp.quicksum(x[i, j, s] for i in range(N + 1) if (i, j, s) in x) == 1)
                model.addConstr(gp.quicksum(x[j, k, s] for k in range(1, N + 2) if (j, k, s) in x) == 1)
                
            model.addConstr(gp.quicksum(x[0, j, s] for j in V_C if (0, j, s) in x) == 
                            gp.quicksum(x[i, N+1, s] for i in V_C if (i, N+1, s) in x))
            
            for i in V_C:
                if (0, i, s) in x:
                    model.addConstr(u[i, s] <= veh_cap - (veh_cap - dems_local[s][i]) * x[0, i, s])
                for j in V_C:
                    if i != j and (i, j, s) in x:
                        model.addConstr(u[j, s] >= u[i, s] + dems_local[s][j] - (veh_cap - dems_local[s][i]) * (1 - x[i, j, s]))
                        
            for i in V_C:
                model.addConstr(t[i, s] >= T_start[i])
                model.addConstr(t[i, s] <= T_start[i] + clients[i]['endo_width'])
                dist_in = dist_matrix[i][-1]
                M_in = clients[i]['exo_end'] - (depot_exo[1] - dist_in)
                if M_in > 0 and (i, N+1, s) in x:
                    model.addConstr(t[i, s] <= depot_exo[1] - dist_in + M_in * (1 - x[i, N+1, s]))

            for i in range(N + 1):
                for j in V_C:
                    if i == j: continue
                    if i == 0:
                        if (0, j, s) in x:
                            dist_0j = dist_matrix[0][j]
                            M_0j = max(0, depot_exo[0] + dist_0j - clients[j]['exo_start'])
                            model.addConstr(t[j, s] >= depot_exo[0] + dist_0j - M_0j * (1 - x[0, j, s]))
                    else:
                        if (i, j, s) in x:
                            dist_ij = dist_matrix[i][j]
                            M_ij = max(0, clients[i]['exo_end'] + dist_ij - clients[j]['exo_start'])
                            model.addConstr(t[j, s] >= t[i, s] + dist_ij - M_ij * (1 - x[i, j, s]))

        if mip_start_sol is not None:
            for var in x.values():
                var.Start = 0.0
            
            for s in range(S_local):
                scenario_id_str = str(scenarios_list[s]['scenario_id'])
                routes_to_use = None
                if scenario_id_str in mip_start_sol["routes"]:
                    routes_to_use = mip_start_sol["routes"][scenario_id_str]
                else:
                    routes_to_use = list(mip_start_sol["routes"].values())[0]
                    
                for r_data in routes_to_use:
                    route_nodes = r_data["route"]
                    for idx in range(len(route_nodes) - 1):
                        u_node = route_nodes[idx]
                        v_node = route_nodes[idx+1]
                        if (u_node, v_node, s) in x:
                            x[u_node, v_node, s].Start = 1.0
                            
                    arr_times = r_data["arrival_times"]
                    for i in V_C:
                        i_str = str(clients[i]['id'])
                        if i_str in arr_times and (i, s) in t:
                            t[i, s].Start = arr_times[i_str]
                            
            for i in V_C:
                i_str = str(clients[i]['id'])
                if i_str in mip_start_sol["time_windows"]:
                    T_start[i].Start = mip_start_sol["time_windows"][i_str][0]

        def extract_solution_inner(mod, solve_cb=False):
            def get_val(var):
                return mod.cbGetSolution(var) if solve_cb else var.X

            time_windows_out = {}
            for i in V_C:
                v_T = get_val(T_start[i])
                w_i = clients[i]['endo_width']
                time_windows_out[str(clients[i]['id'])] = [v_T, v_T + w_i]

            routes_out = {}
            obj_val = 0.0

            for s in range(S_local):
                s_routes = []
                start_nodes = [j for j in V_C if (0, j, s) in x and get_val(x[0, j, s]) > 0.5]
                        
                for first_cust in start_nodes:
                    route = [first_cust]
                    curr = first_cust
                    for _ in range(N + 5):
                        next_cust = None
                        for k in range(1, N + 2):
                            if (curr, k, s) in x and get_val(x[curr, k, s]) > 0.5:
                                next_cust = k
                                break
                        if next_cust is None or next_cust == N + 1:
                            break
                        route.append(next_cust)
                        curr = next_cust
                        
                    arr_times = {}
                    v_t_first = get_val(t[first_cust, s])
                    arr_times["0"] = max(depot_exo[0], v_t_first - dist_matrix[0][first_cust])
                    
                    route_cost = dist_matrix[0][route[0]]
                    for idx, c in enumerate(route):
                        arr_times[str(clients[c]['id'])] = get_val(t[c, s])
                        if idx < len(route) - 1:
                            route_cost += dist_matrix[c][route[idx+1]]
                            
                    last_c = route[-1]
                    arr_times[str(N + 1)] = get_val(t[last_c, s]) + dist_matrix[last_c][-1]
                    route_cost += dist_matrix[last_c][-1]
                    
                    route_nodes = [0] + route + [N + 1]
                    s_routes.append({
                        "route": route_nodes,
                        "arrival_times": arr_times,
                        "cost": route_cost
                    })
                    obj_val += probs_local[s] * route_cost
                    
                routes_out[str(scenarios_list[s]['scenario_id'])] = s_routes
                
            return {
                "objective_value": obj_val,
                "time_windows": time_windows_out,
                "routes": routes_out
            }

        def cb(mod, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    sol = extract_solution_inner(mod, solve_cb=True)
                    if not is_multi:
                        sol = map_single_to_multi(sol, scenarios_data)
                    log_solution(sol)
                except Exception:
                    pass

        model.optimize(cb)
        if model.SolCount > 0:
            sol = extract_solution_inner(model, solve_cb=False)
            return sol
        return None

    dummy_scenario = [{
        "scenario_id": 9999,
        "probability": 1.0,
        "demands": [demands_max[customers_data[j]["id"]] for j in range(num_customers)]
    }]
    
    time_limit_single = min(60, max(10, args.time_limit * 0.15))
    remaining = args.time_limit - (time.time() - start_time_total) - 2
    if remaining > 10:
        single_sol = solve_routing_model(
            dummy_scenario, 
            time_limit_sec=min(time_limit_single, remaining), 
            mip_start_sol=greedy_sol
        )
        if single_sol is not None:
            mapped_sol = map_single_to_multi(single_sol, scenarios_data)
            log_solution(mapped_sol)
            
    remaining = args.time_limit - (time.time() - start_time_total) - 5
    if remaining > 10:
        solve_routing_model(
            scenarios_data,
            time_limit_sec=remaining,
            mip_start_sol=best_sol,
            is_multi=True
        )

    if best_sol is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(best_sol, f, indent=4)

if __name__ == "__main__":
    main()