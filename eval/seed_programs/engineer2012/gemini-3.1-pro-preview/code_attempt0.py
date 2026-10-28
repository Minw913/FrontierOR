import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

class VarStore:
    def __init__(self):
        self.vars = []
        self.keys = []
        self.var_dict = {}
        
    def add(self, key, var):
        self.keys.append(key)
        self.vars.append(var)
        self.var_dict[key] = var
        
    def get_vals(self, model):
        try:
            vals = model.cbGetSolution(self.vars)
            return dict(zip(self.keys, vals))
        except Exception:
            return {}

class CallbackData:
    def __init__(self, logger, z_store, f_store, y_store, q_store, I_store, data):
        self.logger = logger
        self.z_store = z_store
        self.f_store = f_store
        self.y_store = y_store
        self.q_store = q_store
        self.I_store = I_store
        self.data = data

def build_solution(z_vars, f_vars, y_vars, q_vars, I_vals, data, obj):
    sol = {
        "objective_value": obj,
        "vessel_routes": {},
        "port_inventories": {},
        "vessel_operations": {},
        "cost_breakdown": {}
    }
    
    t_cost = 0.0
    d_cost = 0.0
    p_cost = 0.0
    s_rev = 0.0
    
    for vid, v in data['vessels_dict'].items():
        sol['vessel_routes'][vid] = []
        sol['vessel_operations'][vid] = []
        
        # routes
        for a, val in z_vars.get(vid, {}).items():
            if val > 0.5:
                f_val = f_vars[vid][a]
                arc_type = a[0]
                if arc_type == 'source':
                    tail = f"('src', '{vid}')"
                    head = f"('{a[1]}', {a[2]})"
                    cst = 0.0
                    typ = "source"
                elif arc_type == 'sink':
                    tail = f"('{a[1]}', {a[2]})"
                    head = f"('snk', '{vid}')"
                    cst = 0.0
                    typ = "sink"
                elif arc_type == 'demurrage':
                    tail = f"('{a[1]}', {a[2]})"
                    head = f"('{a[1]}', {a[3]})"
                    cst = v['demurrage_cost_per_day']
                    typ = "demurrage"
                    d_cost += cst
                elif arc_type == 'travel':
                    tail = f"('{a[1]}', {a[2]})"
                    head = f"('{a[3]}', {a[4]})"
                    tt = a[4] - a[2]
                    cst = tt * v['transport_cost_per_day']
                    typ = "travel"
                    t_cost += cst
                    
                sol['vessel_routes'][vid].append({
                    "tail": tail,
                    "head": head,
                    "type": typ,
                    "cost": float(cst),
                    "product_flow": float(f_val)
                })
        
        # operations
        for pt, val in y_vars.get(vid, {}).items():
            if val > 0.5:
                p, t = pt
                q_val = q_vars.get(vid, {}).get(pt, 0.0)
                if q_val > 1e-4:
                    if data['ports'][p]['type'] == 'supply':
                        op_type = 'load'
                        qty = float(q_val)
                        p_cost += qty * data['ports'][p]['unit_procurement_cost']
                    else:
                        op_type = 'discharge'
                        qty = -float(q_val)
                        s_rev += float(q_val) * data['ports'][p]['unit_supply_revenue']
                        
                    sol['vessel_operations'][vid].append({
                        "port": p,
                        "time": int(t),
                        "quantity": qty,
                        "type": op_type
                    })
    
    for p in data['all_ports']:
        sol['port_inventories'][p] = []
        for t in range(1, data['T'] + 1):
            val = float(I_vals.get((p, t), 0.0))
            sol['port_inventories'][p].append(val)
            
    sol['cost_breakdown']['transport_cost'] = float(t_cost)
    sol['cost_breakdown']['demurrage_cost'] = float(d_cost)
    sol['cost_breakdown']['procurement_cost'] = float(p_cost)
    sol['cost_breakdown']['supply_revenue'] = float(s_rev)
    
    tot = float(t_cost + d_cost + p_cost - s_rev)
    sol['cost_breakdown']['total_cost'] = tot
    sol['objective_value'] = tot
    
    return sol

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        raw_data = json.load(f)

    T = raw_data['planning_horizon']
    vessels = [v['id'] for v in raw_data['vessels']]
    vessels_dict = {v['id']: v for v in raw_data['vessels']}
    
    supply_ports = [p['port_id'] for p in raw_data['supply_ports']]
    demand_ports = [p['port_id'] for p in raw_data['demand_ports']]
    all_ports = supply_ports + demand_ports
    
    ports_dict = {}
    for p in raw_data['supply_ports']: ports_dict[p['port_id']] = p
    for p in raw_data['demand_ports']: ports_dict[p['port_id']] = p
        
    data = {
        'T': T,
        'vessels': vessels,
        'vessels_dict': vessels_dict,
        'supply_ports': supply_ports,
        'demand_ports': demand_ports,
        'all_ports': all_ports,
        'ports': ports_dict,
        'travel_times': raw_data['travel_times'],
        'draft_limits': raw_data.get('draft_limits', {})
    }
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("Threads", 1)
    env.setParam("TimeLimit", args.time_limit)
    env.start()
    
    m = gp.Model("shipping", env=env)
    
    z_store = VarStore()
    f_store = VarStore()
    y_store = VarStore()
    q_store = VarStore()
    I_store = VarStore()

    for p in all_ports:
        for t in range(1, T + 1):
            cap = ports_dict[p]['storage_capacity'][t-1]
            I_store.add((p, t), m.addVar(lb=0, ub=cap, vtype=GRB.CONTINUOUS))

    arcs_v = {}
    for vid in vessels:
        v = vessels_dict[vid]
        T_S = max(1, v['availability_start'])
        T_E = min(T, v['availability_end'])
        arcs = []
        for p in all_ports:
            arcs.append(('source', p, T_S))
            arcs.append(('sink', p, T_E))
            for t in range(T_S, T_E):
                arcs.append(('demurrage', p, t, t+1))
            for p_to in all_ports:
                if p != p_to:
                    tt = data['travel_times'].get(f"{p}->{p_to}")
                    if tt is not None:
                        for t in range(T_S, T_E + 1):
                            if t + tt <= T_E:
                                arcs.append(('travel', p, t, p_to, t+tt))
        arcs_v[vid] = arcs

        for a in arcs:
            z_var = m.addVar(vtype=GRB.BINARY)
            f_var = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
            z_store.add((vid, a), z_var)
            f_store.add((vid, a), f_var)
            
            p_cap = data['draft_limits'].get(f"{vid}->{a[1]}", v['capacity'])
            max_c = min(v['capacity'], p_cap)
            
            if a[0] == 'travel':
                p_to_cap = data['draft_limits'].get(f"{vid}->{a[3]}", v['capacity'])
                max_c = min(max_c, p_to_cap)
                
            m.addConstr(f_var <= max_c * z_var)
            
            if a[0] == 'source':
                m.addConstr(f_var == v['initial_load'] * z_var)
                
        for p in all_ports:
            for t in range(T_S, T_E + 1):
                y_var = m.addVar(vtype=GRB.BINARY)
                q_var = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
                y_store.add((vid, p, t), y_var)
                q_store.add((vid, p, t), q_var)
                
                F_min = ports_dict[p]['F_min']
                F_max = ports_dict[p]['F_max']
                m.addConstr(q_var >= F_min * y_var)
                m.addConstr(q_var <= F_max * y_var)
                
        m.addConstr(gp.quicksum(z_store.var_dict[(vid, a)] for a in arcs if a[0] == 'source') == 1)
        m.addConstr(gp.quicksum(z_store.var_dict[(vid, a)] for a in arcs if a[0] == 'sink') == 1)
        
        for p in all_ports:
            for t in range(T_S, T_E + 1):
                in_arcs = [a for a in arcs if (a[0] == 'source' and a[1] == p and a[2] == t) or \
                                              (a[0] == 'demurrage' and a[1] == p and a[3] == t) or \
                                              (a[0] == 'travel' and a[3] == p and a[4] == t)]
                out_arcs = [a for a in arcs if (a[0] == 'sink' and a[1] == p and a[2] == t) or \
                                               (a[0] == 'demurrage' and a[1] == p and a[2] == t) or \
                                               (a[0] == 'travel' and a[1] == p and a[2] == t)]
                
                sum_z_in = gp.quicksum(z_store.var_dict[(vid, a)] for a in in_arcs)
                sum_z_out = gp.quicksum(z_store.var_dict[(vid, a)] for a in out_arcs)
                m.addConstr(sum_z_in == sum_z_out)
                m.addConstr(y_store.var_dict[(vid, p, t)] <= sum_z_in)
                
                sum_f_in = gp.quicksum(f_store.var_dict[(vid, a)] for a in in_arcs)
                sum_f_out = gp.quicksum(f_store.var_dict[(vid, a)] for a in out_arcs)
                q_v = q_store.var_dict[(vid, p, t)]
                
                if ports_dict[p]['type'] == 'supply':
                    m.addConstr(sum_f_out - sum_f_in == q_v)
                else:
                    m.addConstr(sum_f_out - sum_f_in == -q_v)

        if 'loading_window_end' in v:
            LWE = v['loading_window_end']
            sum_early = gp.quicksum(y_store.var_dict[(vid, p, t)] 
                                    for p in supply_ports 
                                    for t in range(T_S, LWE + 1) 
                                    if (vid, p, t) in y_store.var_dict)
            for p in supply_ports:
                for t in range(max(T_S, LWE + 1), T_E + 1):
                    if (vid, p, t) in y_store.var_dict:
                        m.addConstr(y_store.var_dict[(vid, p, t)] <= sum_early)
                        
    for p in all_ports:
        port_data = ports_dict[p]
        is_supply = (port_data['type'] == 'supply')
        for t in range(1, T + 1):
            if t == 1:
                prev_I = port_data['initial_inventory']
            else:
                prev_I = I_store.var_dict[(p, t-1)]
                
            sum_q = gp.quicksum(q_store.var_dict[(vid, p, t)] for vid in vessels if (vid, p, t) in q_store.var_dict)
            
            if is_supply:
                rate = port_data['production_rate'][t-1]
                m.addConstr(I_store.var_dict[(p, t)] == prev_I + rate - sum_q)
            else:
                rate = port_data['consumption_rate'][t-1]
                m.addConstr(I_store.var_dict[(p, t)] == prev_I - rate + sum_q)

    obj_expr = gp.LinExpr()
    for vid in vessels:
        v = vessels_dict[vid]
        for a in arcs_v[vid]:
            if a[0] == 'demurrage':
                obj_expr += v['demurrage_cost_per_day'] * z_store.var_dict[(vid, a)]
            elif a[0] == 'travel':
                tt = a[4] - a[2]
                obj_expr += (tt * v['transport_cost_per_day']) * z_store.var_dict[(vid, a)]
                
        for p in all_ports:
            T_S = max(1, v['availability_start'])
            T_E = min(T, v['availability_end'])
            for t in range(T_S, T_E + 1):
                q_var = q_store.var_dict[(vid, p, t)]
                if ports_dict[p]['type'] == 'supply':
                    obj_expr += ports_dict[p]['unit_procurement_cost'] * q_var
                else:
                    obj_expr -= ports_dict[p]['unit_supply_revenue'] * q_var

    m.setObjective(obj_expr, GRB.MINIMIZE)
    
    cb_data = CallbackData(logger, z_store, f_store, y_store, q_store, I_store, data)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if cb_data.logger:
                    z_vals = cb_data.z_store.get_vals(model)
                    f_vals = cb_data.f_store.get_vals(model)
                    y_vals = cb_data.y_store.get_vals(model)
                    q_vals = cb_data.q_store.get_vals(model)
                    I_vals = cb_data.I_store.get_vals(model)
                    
                    z_nested, f_nested, y_nested, q_nested = {}, {}, {}, {}
                    for k, val in z_vals.items(): z_nested.setdefault(k[0], {})[k[1]] = val
                    for k, val in f_vals.items(): f_nested.setdefault(k[0], {})[k[1]] = val
                    for k, val in y_vals.items(): y_nested.setdefault(k[0], {})[(k[1], k[2])] = val
                    for k, val in q_vals.items(): q_nested.setdefault(k[0], {})[(k[1], k[2])] = val
                        
                    sol = build_solution(z_nested, f_nested, y_nested, q_nested, I_vals, cb_data.data, obj)
                    cb_data.logger.log_solution(obj, sol)
            except Exception:
                pass
                
    m.optimize(callback)
    
    if m.SolCount > 0:
        z_vals = {k: v.X for k, v in zip(z_store.keys, z_store.vars)}
        f_vals = {k: v.X for k, v in zip(f_store.keys, f_store.vars)}
        y_vals = {k: v.X for k, v in zip(y_store.keys, y_store.vars)}
        q_vals = {k: v.X for k, v in zip(q_store.keys, q_store.vars)}
        I_vals = {k: v.X for k, v in zip(I_store.keys, I_store.vars)}
        
        z_nested, f_nested, y_nested, q_nested = {}, {}, {}, {}
        for k, v in z_vals.items(): z_nested.setdefault(k[0], {})[k[1]] = v
        for k, v in f_vals.items(): f_nested.setdefault(k[0], {})[k[1]] = v
        for k, v in y_vals.items(): y_nested.setdefault(k[0], {})[(k[1], k[2])] = v
        for k, v in q_vals.items(): q_nested.setdefault(k[0], {})[(k[1], k[2])] = v
            
        final_obj = m.ObjVal
        sol = build_solution(z_nested, f_nested, y_nested, q_nested, I_vals, cb_data.data, final_obj)
        
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
    else:
        sys.exit(1)

if __name__ == "__main__":
    solve()