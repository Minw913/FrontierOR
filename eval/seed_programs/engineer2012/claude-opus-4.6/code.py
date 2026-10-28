import argparse
import json
import math
import time
from itertools import product as iter_product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        data = json.load(f)

    T = data['planning_horizon']
    num_vessels = data['num_vessels']
    vessels = data['vessels']
    supply_ports = data['supply_ports']
    demand_ports = data['demand_ports']
    travel_times_raw = data['travel_times']
    draft_limits_raw = data['draft_limits']

    all_ports = supply_ports + demand_ports
    port_ids = [p['port_id'] for p in all_ports]
    supply_ids = set(p['port_id'] for p in supply_ports)
    demand_ids = set(p['port_id'] for p in demand_ports)
    port_map = {p['port_id']: p for p in all_ports}

    # Parse travel times
    travel_times = {}
    for key, val in travel_times_raw.items():
        parts = key.split('->')
        o, d = parts[0], parts[1]
        travel_times[(o, d)] = val

    # Parse draft limits
    draft_limits = {}
    for key, val in draft_limits_raw.items():
        parts = key.split('->')
        v, p = parts[0], parts[1]
        draft_limits[(v, p)] = val

    import gurobipy as gp
    from gurobipy import GRB

    start_time = time.time()

    m = gp.Model("ship_routing")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, time_limit - 10)  # reserve time for solution extraction

    # Decision variables
    # For each vessel v, we have binary arc variables and continuous flow variables
    
    # Arc types for vessel v:
    # 1. Travel arcs: (port_i, t) -> (port_j, t + travel_time(i,j)) for i != j
    # 2. Demurrage arcs: (port_i, t) -> (port_i, t+1)
    # 3. Source arc: src_v -> (port_i, t) for some starting port/time
    # 4. Sink arc: (port_i, t) -> snk_v
    
    # We need to determine starting location for each vessel
    # The problem says vessel has availability_start and availability_end
    # We need to figure out where vessels start - looking at the data, vessels might start at any port
    # Actually, the problem doesn't specify starting port explicitly in the schema
    # Let me re-read: "the vessel begins with a specified initial load on board"
    # The source/sink are virtual nodes - vessel can start at any port at availability_start
    # and end at any port at availability_end
    
    # Build arcs for each vessel
    vessel_arcs = {}  # vessel_id -> list of (tail, head, type, cost)
    arc_vars = {}  # (vessel_id, arc_idx) -> binary var
    flow_vars = {}  # (vessel_id, arc_idx) -> continuous var (cargo on board)
    
    # Load/discharge variables at (vessel, port, time)
    load_vars = {}  # (vessel_id, port_id, t) -> continuous var (amount loaded/discharged)
    load_binary = {}  # (vessel_id, port_id, t) -> binary var (whether operation happens)
    
    # Port inventory variables
    inv_vars = {}  # (port_id, t) -> continuous var
    
    # Create inventory variables for all ports and times 1..T
    for p in all_ports:
        pid = p['port_id']
        for t in range(1, T + 1):
            cap = p['storage_capacity'][t - 1] if 'storage_capacity' in p else float('inf')
            inv_vars[(pid, t)] = m.addVar(lb=0, ub=cap, name=f"inv_{pid}_{t}")

    # For each vessel, build time-space network
    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        cap = v['capacity']
        tc = v['transport_cost_per_day']
        dc = v['demurrage_cost_per_day']
        init_load = v['initial_load']

        arcs = []
        
        # Source arcs: from source to any (port, t_start)
        # Actually, vessel could potentially start at t_start at any port
        for p in port_ids:
            dl = draft_limits.get((vid, p), cap)
            ub = min(cap, dl)
            arcs.append((f"(src, {vid})", f"({p}, {t_start})", 'source', 0, ub))
        
        # Sink arcs: from any (port, t) to sink for t in [t_start, t_end]
        for p in port_ids:
            for t in range(t_start, t_end + 1):
                dl = draft_limits.get((vid, p), cap)
                ub = min(cap, dl)
                arcs.append((f"({p}, {t})", f"(snk, {vid})", 'sink', 0, ub))
        
        # Demurrage arcs: (p, t) -> (p, t+1)
        for p in port_ids:
            for t in range(t_start, t_end):
                dl = draft_limits.get((vid, p), cap)
                ub = min(cap, dl)
                arcs.append((f"({p}, {t})", f"({p}, {t+1})", 'demurrage', dc, ub))
        
        # Travel arcs: (p1, t) -> (p2, t + travel_time)
        for p1 in port_ids:
            for p2 in port_ids:
                if p1 == p2:
                    continue
                tt = travel_times.get((p1, p2), None)
                if tt is None:
                    continue
                for t in range(t_start, t_end + 1):
                    t_arr = t + tt
                    if t_arr > t_end:
                        continue
                    dl_dest = draft_limits.get((vid, p2), cap)
                    dl_orig = draft_limits.get((vid, p1), cap)
                    ub = min(cap, dl_dest, dl_orig)
                    arcs.append((f"({p1}, {t})", f"({p2}, {t_arr})", 'travel', tc * tt, ub))
        
        vessel_arcs[vid] = arcs
        
        # Create variables for each arc
        for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
            arc_vars[(vid, idx)] = m.addVar(vtype=GRB.BINARY, name=f"x_{vid}_{idx}")
            flow_vars[(vid, idx)] = m.addVar(lb=0, ub=ub, name=f"f_{vid}_{idx}")

    # Create load/discharge variables
    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        cap = v['capacity']
        
        for p in all_ports:
            pid = p['port_id']
            if pid in supply_ids:
                fmin = p['F_min']
                fmax = p['F_max']
            else:
                fmin = p['F_min']
                fmax = p['F_max']
            
            for t in range(t_start, t_end + 1):
                load_vars[(vid, pid, t)] = m.addVar(lb=0, ub=fmax, name=f"q_{vid}_{pid}_{t}")
                load_binary[(vid, pid, t)] = m.addVar(vtype=GRB.BINARY, name=f"b_{vid}_{pid}_{t}")

    m.update()

    # Helper: parse node to get (port, time) or None for src/snk
    def parse_node(s):
        s = s.strip()
        if s.startswith("(src") or s.startswith("(snk"):
            return None
        # Format: (port_id, time)
        inner = s[1:-1]  # remove parens
        parts = inner.rsplit(',', 1)
        port = parts[0].strip()
        t = int(parts[1].strip())
        return (port, t)

    # Flow conservation constraints for each vessel
    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        cap = v['capacity']
        init_load = v['initial_load']
        arcs = vessel_arcs[vid]
        
        # Collect all nodes
        nodes = set()
        for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
            nodes.add(tail)
            nodes.add(head)
        
        # For each node, flow conservation of arc selection (binary)
        # Source: out - in = 1
        # Sink: in - out = 1
        # Others: in = out
        
        src_node = f"(src, {vid})"
        snk_node = f"(snk, {vid})"
        
        for node in nodes:
            out_arcs = [idx for idx, (tail, head, atype, cost, ub) in enumerate(arcs) if tail == node]
            in_arcs = [idx for idx, (tail, head, atype, cost, ub) in enumerate(arcs) if head == node]
            
            if node == src_node:
                m.addConstr(
                    gp.quicksum(arc_vars[(vid, idx)] for idx in out_arcs) -
                    gp.quicksum(arc_vars[(vid, idx)] for idx in in_arcs) == 1,
                    name=f"flow_src_{vid}"
                )
            elif node == snk_node:
                m.addConstr(
                    gp.quicksum(arc_vars[(vid, idx)] for idx in in_arcs) -
                    gp.quicksum(arc_vars[(vid, idx)] for idx in out_arcs) == 1,
                    name=f"flow_snk_{vid}"
                )
            else:
                m.addConstr(
                    gp.quicksum(arc_vars[(vid, idx)] for idx in out_arcs) -
                    gp.quicksum(arc_vars[(vid, idx)] for idx in in_arcs) == 0,
                    name=f"flow_cons_{vid}_{node}"
                )
        
        # Cargo flow conservation at each port-time node:
        # For a port-time node (p, t):
        #   sum of flow on outgoing arcs - sum of flow on incoming arcs = load_amount at (v, p, t)
        #   where load_amount is positive for loading (supply) and negative for discharging (demand)
        # But we need to be careful: flow on arc = cargo on board while traversing that arc
        
        # At node (p, t): flow_out = flow_in + loaded - discharged
        # loaded at supply port, discharged at demand port
        
        for node in nodes:
            parsed = parse_node(node)
            if parsed is None:
                continue
            port, t_node = parsed
            
            out_arcs = [idx for idx, (tail, head, atype, cost, ub) in enumerate(arcs) if tail == node]
            in_arcs = [idx for idx, (tail, head, atype, cost, ub) in enumerate(arcs) if head == node]
            
            # Vessel presence: can only load/discharge if vessel is at this node
            # Vessel is present at (p,t) if there's an outgoing arc selected from this node
            presence = gp.quicksum(arc_vars[(vid, idx)] for idx in out_arcs)
            
            # Also incoming presence (for nodes that are not source-connected)
            # Actually for source arcs, the head is (p, t_start), and for those the incoming is from src
            
            if (vid, port, t_node) in load_binary:
                # Load/discharge can only happen if vessel is present
                m.addConstr(load_binary[(vid, port, t_node)] <= presence,
                           name=f"presence_{vid}_{port}_{t_node}")
                
                # Quantity bounds
                p_data = port_map[port]
                if port in supply_ids:
                    fmin = p_data['F_min']
                    fmax = p_data['F_max']
                else:
                    fmin = p_data['F_min']
                    fmax = p_data['F_max']
                
                m.addConstr(load_vars[(vid, port, t_node)] >= fmin * load_binary[(vid, port, t_node)],
                           name=f"qmin_{vid}_{port}_{t_node}")
                m.addConstr(load_vars[(vid, port, t_node)] <= fmax * load_binary[(vid, port, t_node)],
                           name=f"qmax_{vid}_{port}_{t_node}")
            
            # Cargo flow balance at this node
            # flow leaving = flow entering + net_load
            # For supply port: net_load = +load_vars (loading onto vessel)
            # For demand port: net_load = -load_vars (discharging from vessel)
            
            if (vid, port, t_node) in load_vars:
                if port in supply_ids:
                    net_load = load_vars[(vid, port, t_node)]
                else:
                    net_load = -load_vars[(vid, port, t_node)]
            else:
                net_load = 0
            
            # sum of flow on outgoing arcs = sum of flow on incoming arcs + net_load
            # But we need to handle that flow_vars are only defined when arc_vars = 1
            # flow_vars[(vid, idx)] <= ub * arc_vars[(vid, idx)]
            
            m.addConstr(
                gp.quicksum(flow_vars[(vid, idx)] for idx in out_arcs) ==
                gp.quicksum(flow_vars[(vid, idx)] for idx in in_arcs) + net_load,
                name=f"cargo_balance_{vid}_{port}_{t_node}"
            )
        
        # Flow on source arcs must equal initial_load when arc is selected
        # flow on source arc = init_load * arc_var
        for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
            if atype == 'source':
                m.addConstr(flow_vars[(vid, idx)] == init_load * arc_vars[(vid, idx)],
                           name=f"init_load_{vid}_{idx}")
        
        # Link flow and arc binary: flow <= ub * arc_var
        for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
            m.addConstr(flow_vars[(vid, idx)] <= ub * arc_vars[(vid, idx)],
                       name=f"flow_link_{vid}_{idx}")

    # Port inventory constraints
    for p in supply_ports:
        pid = p['port_id']
        init_inv = p['initial_inventory']
        prod_rate = p['production_rate']
        
        for t in range(1, T + 1):
            total_loaded = gp.quicksum(
                load_vars[(v['id'], pid, t)]
                for v in vessels
                if (v['id'], pid, t) in load_vars
            )
            
            if t == 1:
                prev_inv = init_inv
            else:
                prev_inv = inv_vars[(pid, t - 1)]
            
            m.addConstr(
                inv_vars[(pid, t)] == prev_inv + prod_rate[t - 1] - total_loaded,
                name=f"inv_supply_{pid}_{t}"
            )

    for p in demand_ports:
        pid = p['port_id']
        init_inv = p['initial_inventory']
        cons_rate = p['consumption_rate']
        
        for t in range(1, T + 1):
            total_discharged = gp.quicksum(
                load_vars[(v['id'], pid, t)]
                for v in vessels
                if (v['id'], pid, t) in load_vars
            )
            
            if t == 1:
                prev_inv = init_inv
            else:
                prev_inv = inv_vars[(pid, t - 1)]
            
            m.addConstr(
                inv_vars[(pid, t)] == prev_inv - cons_rate[t - 1] + total_discharged,
                name=f"inv_demand_{pid}_{t}"
            )

    # Objective: minimize total cost
    # Cost = transport + demurrage + procurement - revenue
    obj = gp.LinExpr()
    
    for v in vessels:
        vid = v['id']
        arcs = vessel_arcs[vid]
        
        for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
            obj += cost * arc_vars[(vid, idx)]
    
    # Procurement cost at supply ports
    for p in supply_ports:
        pid = p['port_id']
        unit_cost = p['unit_procurement_cost']
        for v in vessels:
            vid = v['id']
            for t in range(1, T + 1):
                if (vid, pid, t) in load_vars:
                    obj += unit_cost * load_vars[(vid, pid, t)]
    
    # Revenue at demand ports (subtract from cost)
    for p in demand_ports:
        pid = p['port_id']
        unit_rev = p['unit_supply_revenue']
        for v in vessels:
            vid = v['id']
            for t in range(1, T + 1):
                if (vid, pid, t) in load_vars:
                    obj -= unit_rev * load_vars[(vid, pid, t)]
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Store best solution for callback
    best_solution = [None]
    best_obj = [float('inf')]
    
    def extract_solution():
        """Extract solution from current Gurobi solution."""
        try:
            sol = {}
            sol['objective_value'] = m.ObjVal
            
            # Vessel routes
            vessel_routes = {}
            for v in vessels:
                vid = v['id']
                arcs = vessel_arcs[vid]
                route = []
                for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
                    if arc_vars[(vid, idx)].X > 0.5:
                        route.append({
                            'tail': tail,
                            'head': head,
                            'type': atype,
                            'cost': cost,
                            'product_flow': round(flow_vars[(vid, idx)].X, 6)
                        })
                vessel_routes[vid] = route
            sol['vessel_routes'] = vessel_routes
            
            # Port inventories
            port_inventories = {}
            for p in all_ports:
                pid = p['port_id']
                inv_list = {}
                for t in range(1, T + 1):
                    inv_list[str(t)] = round(inv_vars[(pid, t)].X, 6)
                port_inventories[pid] = inv_list
            sol['port_inventories'] = port_inventories
            
            # Vessel operations
            vessel_operations = {}
            for v in vessels:
                vid = v['id']
                ops = []
                for p in all_ports:
                    pid = p['port_id']
                    for t in range(1, T + 1):
                        if (vid, pid, t) in load_vars:
                            qty = load_vars[(vid, pid, t)].X
                            if qty > 1e-6:
                                if pid in supply_ids:
                                    ops.append({
                                        'port': pid,
                                        'time': t,
                                        'quantity': round(qty, 6),
                                        'type': 'load'
                                    })
                                else:
                                    ops.append({
                                        'port': pid,
                                        'time': t,
                                        'quantity': round(-qty, 6),
                                        'type': 'discharge'
                                    })
                vessel_operations[vid] = ops
            sol['vessel_operations'] = vessel_operations
            
            # Cost breakdown
            transport_cost = 0
            demurrage_cost = 0
            procurement_cost = 0
            supply_revenue = 0
            
            for v in vessels:
                vid = v['id']
                arcs = vessel_arcs[vid]
                for idx, (tail, head, atype, cost, ub) in enumerate(arcs):
                    if arc_vars[(vid, idx)].X > 0.5:
                        if atype == 'travel':
                            transport_cost += cost
                        elif atype == 'demurrage':
                            demurrage_cost += cost
            
            for p in supply_ports:
                pid = p['port_id']
                unit_cost = p['unit_procurement_cost']
                for v in vessels:
                    vid = v['id']
                    for t in range(1, T + 1):
                        if (vid, pid, t) in load_vars:
                            qty = load_vars[(vid, pid, t)].X
                            if qty > 1e-6:
                                procurement_cost += unit_cost * qty
            
            for p in demand_ports:
                pid = p['port_id']
                unit_rev = p['unit_supply_revenue']
                for v in vessels:
                    vid = v['id']
                    for t in range(1, T + 1):
                        if (vid, pid, t) in load_vars:
                            qty = load_vars[(vid, pid, t)].X
                            if qty > 1e-6:
                                supply_revenue += unit_rev * qty
            
            total_cost = transport_cost + demurrage_cost + procurement_cost - supply_revenue
            
            sol['cost_breakdown'] = {
                'transport_cost': round(transport_cost, 6),
                'demurrage_cost': round(demurrage_cost, 6),
                'procurement_cost': round(procurement_cost, 6),
                'supply_revenue': round(supply_revenue, 6),
                'total_cost': round(total_cost, 6)
            }
            
            return sol
        except Exception as e:
            return None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    m.optimize(callback)
    
    sol = None
    if m.SolCount > 0:
        sol = extract_solution()
        if sol and logger:
            logger.log_solution(sol['objective_value'], sol)
    
    if sol is None:
        # Return empty/default solution
        sol = {
            'objective_value': 0,
            'vessel_routes': {v['id']: [] for v in vessels},
            'port_inventories': {p['port_id']: {str(t): 0 for t in range(1, T+1)} for p in all_ports},
            'vessel_operations': {v['id']: [] for v in vessels},
            'cost_breakdown': {
                'transport_cost': 0,
                'demurrage_cost': 0,
                'procurement_cost': 0,
                'supply_revenue': 0,
                'total_cost': 0
            }
        }
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)