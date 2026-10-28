import argparse
import json
import logging
import sys
import gurobipy as gp
from gurobipy import GRB

# Attempt import of SolutionLogger
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


class MipLogger:
    def __init__(self, logger):
        self.logger = logger
        self.best_obj = float('inf')

    def callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < self.best_obj - 1e-5:
                self.best_obj = obj
                if self.logger:
                    self.logger.log(obj)


def get_dict_val(d, float_key):
    """Safely extracts dictionary value when keys are floats represented as strings."""
    str_key = str(float_key)
    if str_key in d:
        return d[str_key]
    for k, v in d.items():
        try:
            if abs(float(k) - float(float_key)) < 1e-5:
                return v
        except ValueError:
            pass
    raise KeyError(f"Key {float_key} not found in dictionary.")


def get_distance(distances, port1, port2):
    """Gets distance safely considering direction naming conventions."""
    key1 = f"{port1}_{port2}"
    if key1 in distances:
        return float(distances[key1])
    key2 = f"{port2}_{port1}"
    if key2 in distances:
        return float(distances[key2])
    return 0.0


def build_and_solve_model(args, data, logger):
    model = gp.Model("BerthAllocation")
    
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
        
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.2)
    model.setParam('OutputFlag', 1)

    ports = {p["id"]: p for p in data["ports"]}
    distances = data["distances_nautical_miles"]
    ships = data["ships"]
    external_ships = data["external_ships"]
    beta = float(data["handling_time_deviation_factor_beta_per_meter"])
    
    cp = data["cost_parameters"]
    fuel_price = float(cp["fuel_price_usd_per_tonne"])
    hand_rate = float(cp["handling_cost_usd_per_hour"])
    delay_rate = float(cp["delay_cost_usd_per_hour"])
    wait_rate = float(cp["waiting_cost_usd_per_hour"])
    lft_penalty = float(cp["lft_penalty_usd"])
    
    speeds = data["speeds_knots"]
    speed_travel_times = [get_dict_val(data["travel_time_hours_per_nm_by_speed"], sp) for sp in speeds]

    # Variables structure
    X, T, A, H, W, D, O, d_var, Speed = {}, {}, {}, {}, {}, {}, {}, {}, {}
    
    obj_expr = 0
    
    # Building individual ship constraints
    for s_idx, ship in enumerate(ships):
        length = float(ship["length_meters"])
        
        for v_idx, visit in enumerate(ship["port_visits"]):
            port_id = visit["port_id"]
            quay_len = float(ports[port_id]["quay_length_meters"])
            
            X[s_idx, v_idx] = model.addVar(lb=0, ub=max(0.0, quay_len - length), name=f"X_{s_idx}_{v_idx}")
            T[s_idx, v_idx] = model.addVar(lb=float(visit["earliest_start_time_hours"]), name=f"T_{s_idx}_{v_idx}")
            A[s_idx, v_idx] = model.addVar(lb=0, name=f"A_{s_idx}_{v_idx}")
            H[s_idx, v_idx] = model.addVar(lb=float(visit["min_handling_time_hours"]), 
                                           ub=float(visit["max_handling_time_hours"]), 
                                           name=f"H_{s_idx}_{v_idx}")
            W[s_idx, v_idx] = model.addVar(lb=0, name=f"W_{s_idx}_{v_idx}")
            D[s_idx, v_idx] = model.addVar(lb=0, name=f"D_{s_idx}_{v_idx}")
            O[s_idx, v_idx] = model.addVar(lb=0, name=f"O_{s_idx}_{v_idx}")
            d_var[s_idx, v_idx] = model.addVar(lb=0, ub=quay_len, name=f"d_{s_idx}_{v_idx}")
            
            ideal_pos = float(visit["ideal_berthing_position"])
            
            # Position bounds calculation
            model.addConstr(d_var[s_idx, v_idx] >= X[s_idx, v_idx] - ideal_pos)
            model.addConstr(d_var[s_idx, v_idx] >= ideal_pos - X[s_idx, v_idx])
            
            # Handling time calculation scaled by deviation
            min_H = float(visit["min_handling_time_hours"])
            model.addConstr(H[s_idx, v_idx] == min_H * (1.0 + beta * d_var[s_idx, v_idx]))
            
            # Waiting time relationship
            model.addConstr(W[s_idx, v_idx] == T[s_idx, v_idx] - A[s_idx, v_idx])
            
            # Delay and LFT overshoots
            exp_finish = float(visit["expected_finish_time_hours"])
            model.addConstr(D[s_idx, v_idx] >= T[s_idx, v_idx] + H[s_idx, v_idx] - exp_finish)
            
            lft = float(visit["latest_finish_time_hours"])
            model.addConstr(O[s_idx, v_idx] >= T[s_idx, v_idx] + H[s_idx, v_idx] - lft)
            
            # Formulating objective penalties for scheduling
            obj_expr += wait_rate * W[s_idx, v_idx]
            obj_expr += hand_rate * H[s_idx, v_idx]
            obj_expr += delay_rate * D[s_idx, v_idx]
            obj_expr += lft_penalty * O[s_idx, v_idx]
            
            # Voyage legs mapping constraints
            if v_idx < len(ship["port_visits"]) - 1:
                next_port_id = ship["port_visits"][v_idx+1]["port_id"]
                dist = get_distance(distances, port_id, next_port_id)
                
                speed_terms = []
                fuel_terms = []
                
                for sp_idx, sp in enumerate(speeds):
                    Speed[s_idx, v_idx, sp_idx] = model.addVar(vtype=GRB.BINARY, name=f"Sp_{s_idx}_{v_idx}_{sp_idx}")
                    speed_terms.append(Speed[s_idx, v_idx, sp_idx])
                    
                    travel_time = dist * speed_travel_times[sp_idx]
                    fuel_per_nm = get_dict_val(ship["fuel_consumption_per_nm_by_speed"], sp)
                    cost = fuel_price * fuel_per_nm * dist
                    fuel_terms.append(cost * Speed[s_idx, v_idx, sp_idx])
                
                model.addConstr(gp.quicksum(speed_terms) == 1)
                
                travel_time_expr = gp.quicksum(Speed[s_idx, v_idx, sp_idx] * (dist * speed_travel_times[sp_idx]) for sp_idx in range(len(speeds)))
                model.addConstr(A[s_idx, v_idx+1] == T[s_idx, v_idx] + H[s_idx, v_idx] + travel_time_expr)
                
                obj_expr += gp.quicksum(fuel_terms)

    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Overlap resolution (Grouping calls by port bounds)
    visits_by_port = {port["id"]: {"opt": [], "ext": []} for port in data["ports"]}

    for s_idx, ship in enumerate(ships):
        length = float(ship["length_meters"])
        for v_idx, visit in enumerate(ship["port_visits"]):
            visits_by_port[visit["port_id"]]["opt"].append({
                "s_idx": s_idx, "v_idx": v_idx, "length": length
            })
            
    for ext_group in external_ships:
        port_id = ext_group["port_id"]
        for ext in ext_group["ships"]:
            visits_by_port[port_id]["ext"].append(ext)
            
    M_time = 100000.0  # Big-M indicator scaling safely covering planning schedules

    for port_id, visits in visits_by_port.items():
        opt = visits["opt"]
        ext = visits["ext"]
        quay_len = float(ports[port_id]["quay_length_meters"])
        
        # Internally bounding opt-to-opt overlaps geometrically/temporally
        for i in range(len(opt)):
            s1, v1_i, l1 = opt[i]["s_idx"], opt[i]["v_idx"], opt[i]["length"]
            for j in range(i + 1, len(opt)):
                s2, v2_i, l2 = opt[j]["s_idx"], opt[j]["v_idx"], opt[j]["length"]
                
                z1 = model.addVar(vtype=GRB.BINARY, name=f"Zoo1_{port_id}_{i}x{j}")
                z2 = model.addVar(vtype=GRB.BINARY, name=f"Zoo2_{port_id}_{i}x{j}")
                z3 = model.addVar(vtype=GRB.BINARY, name=f"Zoo3_{port_id}_{i}x{j}")
                z4 = model.addVar(vtype=GRB.BINARY, name=f"Zoo4_{port_id}_{i}x{j}")
                
                model.addConstr(z1 + z2 + z3 + z4 >= 1)
                
                model.addConstr(X[s1, v1_i] + l1 <= X[s2, v2_i] + quay_len * (1 - z1))
                model.addConstr(X[s2, v2_i] + l2 <= X[s1, v1_i] + quay_len * (1 - z2))
                model.addConstr(T[s1, v1_i] + H[s1, v1_i] <= T[s2, v2_i] + M_time * (1 - z3))
                model.addConstr(T[s2, v2_i] + H[s2, v2_i] <= T[s1, v1_i] + M_time * (1 - z4))
                
        # Bounding external fixed bounds sequentially
        for i in range(len(opt)):
            s1, v1_i, l1 = opt[i]["s_idx"], opt[i]["v_idx"], opt[i]["length"]
            for j in range(len(ext)):
                fixed = ext[j]
                e_X, e_L = float(fixed["berthing_position"]), float(fixed["length_meters"])
                e_T, e_END = float(fixed["start_time_hours"]), float(fixed["end_time_hours"])
                
                z1 = model.addVar(vtype=GRB.BINARY, name=f"Zoe1_{port_id}_{i}x{j}")
                z2 = model.addVar(vtype=GRB.BINARY, name=f"Zoe2_{port_id}_{i}x{j}")
                z3 = model.addVar(vtype=GRB.BINARY, name=f"Zoe3_{port_id}_{i}x{j}")
                z4 = model.addVar(vtype=GRB.BINARY, name=f"Zoe4_{port_id}_{i}x{j}")
                
                model.addConstr(z1 + z2 + z3 + z4 >= 1)
                
                model.addConstr(X[s1, v1_i] + l1 <= e_X + quay_len * (1 - z1))
                model.addConstr(e_X + e_L <= X[s1, v1_i] + quay_len * (1 - z2))
                model.addConstr(T[s1, v1_i] + H[s1, v1_i] <= e_T + M_time * (1 - z3))
                model.addConstr(e_END <= T[s1, v1_i] + M_time * (1 - z4))

    mip_logger = MipLogger(logger)
    model.optimize(mip_logger.callback)
    
    if model.SolCount > 0:
        solution = {
            "objective_value": float(model.ObjVal),
            "ships": []
        }
        
        for s_idx, ship in enumerate(ships):
            ship_out = {
                "ship_id": ship["id"],
                "port_visits": [],
                "speed_selections": []
            }
            
            for v_idx, visit in enumerate(ship["port_visits"]):
                pos = max(0.0, float(X[s_idx, v_idx].X))
                start = float(T[s_idx, v_idx].X)
                hand = float(H[s_idx, v_idx].X)
                arr = float(A[s_idx, v_idx].X)
                del_ = max(0.0, float(D[s_idx, v_idx].X))
                overshoot = max(0.0, float(O[s_idx, v_idx].X))
                pos_dev = max(0.0, float(d_var[s_idx, v_idx].X))
                
                ship_out["port_visits"].append({
                    "port_call_index": visit["port_call_index"],
                    "port_id": visit["port_id"],
                    "berthing_position": round(pos, 6),
                    "start_time": round(start, 6),
                    "handling_time": round(hand, 6),
                    "arrival_time": round(arr, 6),
                    "delay": round(del_, 6),
                    "lft_violation": round(overshoot, 6),
                    "position_deviation": round(pos_dev, 6)
                })
                
                if v_idx < len(ship["port_visits"]) - 1:
                    selected_speed = float(speeds[0])
                    for sp_idx, sp in enumerate(speeds):
                        if float(Speed[s_idx, v_idx, sp_idx].X) > 0.5:
                            selected_speed = float(sp)
                            break
                            
                    ship_out["speed_selections"].append({
                        "from_call": visit["port_call_index"],
                        "to_call": ship["port_visits"][v_idx+1]["port_call_index"],
                        "speed_knots": selected_speed
                    })
                    
            solution["ships"].append(ship_out)
            
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
            
    else:
        print("No feasible solution was found within the time limit.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    logger = None
    if SolutionLogger is not None and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    build_and_solve_model(args, instance_data, logger)