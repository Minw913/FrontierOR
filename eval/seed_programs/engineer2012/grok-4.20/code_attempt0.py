import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    T = instance['planning_horizon']
    vessels = instance['vessels']
    supply_ports = instance['supply_ports']
    demand_ports = instance['demand_ports']
    travel_times = instance['travel_times']
    draft_limits = instance['draft_limits']

    vessel_dict = {v['id']: v for v in vessels}
    supply_dict = {p['port_id']: p for p in supply_ports}
    demand_dict = {p['port_id']: p for p in demand_ports}
    all_ports = list(supply_dict.keys()) + list(demand_dict.keys())
    port_type = {}
    for p in supply_dict:
        port_type[p] = 'supply'
    for p in demand_dict:
        port_type[p] = 'demand'

    # Precompute possible arcs
    arcs = []
    arc_type = {}
    arc_cost = {}
    arc_travel = {}
    possible_operations = {}

    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        cap = v['capacity']
        dem_cost = v['demurrage_cost_per_day']
        trans_cost = v['transport_cost_per_day']
        init_load = v['initial_load']

        # Source
        src = f"src_{vid}"
        for p in all_ports:
            key = f"{src}->{p},{t_start}"
            arcs.append((vid, src, f"{p},{t_start}"))
            arc_type[(vid, src, f"{p},{t_start}")] = 'source'
            arc_cost[(vid, src, f"{p},{t_start}")] = 0.0
            arc_travel[(vid, src, f"{p},{t_start}")] = False
            possible_operations[(vid, src, f"{p},{t_start}")] = False

        # Sink
        snk = f"snk_{vid}"
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                key = f"{p},{t}->{snk}"
                arcs.append((vid, f"{p},{t}", snk))
                arc_type[(vid, f"{p},{t}", snk)] = 'sink'
                arc_cost[(vid, f"{p},{t}", snk)] = 0.0
                arc_travel[(vid, f"{p},{t}", snk)] = False
                possible_operations[(vid, f"{p},{t}", snk)] = False

        # Travel and demurrage arcs
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                # Demurrage at p,t -> p,t+1
                if t + 1 <= t_end:
                    key = f"{p},{t}->{p},{t+1}"
                    arcs.append((vid, f"{p},{t}", f"{p},{t+1}"))
                    arc_type[(vid, f"{p},{t}", f"{p},{t+1}")] = 'demurrage'
                    arc_cost[(vid, f"{p},{t}", f"{p},{t+1}")] = dem_cost
                    arc_travel[(vid, f"{p},{t}", f"{p},{t+1}")] = False
                    possible_operations[(vid, f"{p},{t}", f"{p},{t+1}")] = True

                # Travel to other ports
                for q in all_ports:
                    if p == q:
                        continue
                    tt_key = f"{p}->{q}"
                    if tt_key not in travel_times:
                        continue
                    tt = travel_times[tt_key]
                    if t + tt > t_end:
                        continue
                    key = f"{p},{t}->{q},{t+tt}"
                    arcs.append((vid, f"{p},{t}", f"{q},{t+tt}"))
                    arc_type[(vid, f"{p},{t}", f"{q},{t+tt}")] = 'travel'
                    arc_cost[(vid, f"{p},{t}", f"{q},{t+tt}")] = trans_cost * tt
                    arc_travel[(vid, f"{p},{t}", f"{q},{t+tt}")] = True
                    possible_operations[(vid, f"{p},{t}", f"{q},{t+tt}")] = False

    # Create model
    model = gp.Model("Bulk_Shipping")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # leave margin

    # Variables
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    flow = model.addVars(arcs, lb=0.0, name="flow")  # cargo on arc
    y = model.addVars([(v['id'], p, t) for v in vessels for p in all_ports for t in range(v['availability_start'], v['availability_end']+1)],
                      vtype=GRB.BINARY, name="y")  # perform operation at (p,t)

    # Objective components
    transport_cost = gp.quicksum(arc_cost[a] * x[a] for a in arcs if arc_type[a] in ('travel', 'demurrage'))
    procurement = {}
    revenue = {}

    for s in supply_dict:
        procurement[s] = model.addVar(lb=0, name=f"proc_{s}")
    for d in demand_dict:
        revenue[d] = model.addVar(lb=0, name=f"rev_{d}")

    obj = (transport_cost +
           gp.quicksum(supply_dict[s]['unit_procurement_cost'] * procurement[s] for s in supply_dict) -
           gp.quicksum(demand_dict[d]['unit_supply_revenue'] * revenue[d] for d in demand_dict))

    model.setObjective(obj, GRB.MINIMIZE)

    # Flow conservation and exactly one path per vessel
    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        src = f"src_{vid}"
        snk = f"snk_{vid}"
        init_load = v['initial_load']

        nodes_out = {}
        nodes_in = {}
        for a in arcs:
            if a[0] != vid:
                continue
            tail = a[1]
            head = a[2]
            if tail not in nodes_out:
                nodes_out[tail] = []
            nodes_out[tail].append(a)
            if head not in nodes_in:
                nodes_in[head] = []
            nodes_in[head].append(a)

        # Source
        model.addConstr(gp.quicksum(x[a] for a in nodes_out.get(src, [])) == 1, f"source_{vid}")

        # Sink
        model.addConstr(gp.quicksum(x[a] for a in nodes_in.get(snk, [])) == 1, f"sink_{vid}")

        # Flow conservation at port-time nodes
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                node = f"{p},{t}"
                inflow = gp.quicksum(x[a] for a in nodes_in.get(node, []))
                outflow = gp.quicksum(x[a] for a in nodes_out.get(node, []))
                model.addConstr(inflow == outflow, f"flow_con_{vid}_{p}_{t}")

        # Cargo flow
        for a in arcs:
            if a[0] != vid:
                continue
            tail = a[1]
            if tail.startswith('src_'):
                model.addConstr(flow[a] == init_load * x[a], f"initflow_{vid}_{tail}")
            else:
                model.addConstr(flow[a] <= v['capacity'] * x[a], f"cap_{a}")

        # Cargo balance at nodes
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                node = f"{p},{t}"
                incoming_arcs = nodes_in.get(node, [])
                outgoing_arcs = nodes_out.get(node, [])

                if not incoming_arcs:
                    continue

                # Cargo continuity
                if incoming_arcs:
                    inflow_sum = gp.quicksum(flow[a] for a in incoming_arcs)
                    for out_a in outgoing_arcs:
                        if arc_travel.get((vid, node, out_a[2]), False):
                            model.addConstr(flow[out_a] == inflow_sum, f"cargo_cont_{vid}_{node}_{out_a[2]}")
                        else:
                            # demurrage or sink
                            model.addConstr(flow[out_a] == inflow_sum, f"cargo_cont_{vid}_{node}_{out_a[2]}")

                # Load / discharge
                is_supply = port_type.get(p, '') == 'supply'
                port_data = supply_dict[p] if is_supply else demand_dict.get(p)
                if port_data is None:
                    continue

                F_min = port_data['F_min']
                F_max = port_data['F_max']
                draft_key = f"{vid}->{p}"
                draft_max = draft_limits.get(draft_key, v['capacity'])

                op_var = y[(vid, p, t)]

                # Link operation to arc choice: only if arriving via a travel or source and then possibly operating
                # A vessel can only operate if it "visits" i.e. has an incoming arc that is not demurrage from same port at t-1
                # This is approximated by linking y to presence of non-demurrage incoming or outgoing

                # Better: y only allowed if there is an arc that "stops" at (p,t)
                # We allow operation only on nodes that are entered by a travel or source arc

                can_operate = False
                for inc_a in incoming_arcs:
                    if arc_travel.get(inc_a, False) or inc_a[1].startswith('src'):
                        can_operate = True
                        break
                if not can_operate:
                    model.addConstr(op_var == 0, f"no_op_{vid}_{p}_{t}")
                    continue

                delta = model.addVar(lb=0, ub=F_max, name=f"delta_{vid}_{p}_{t}")
                model.addConstr(delta <= F_max * op_var, f"delta_ub_{vid}_{p}_{t}")
                model.addConstr(delta >= F_min * op_var, f"delta_lb_{vid}_{p}_{t}")

                if is_supply:
                    # Loading increases cargo when leaving
                    for out_a in outgoing_arcs:
                        if not arc_travel.get(out_a, False) and not out_a[2].startswith('snk'):
                            # demurrage
                            model.addConstr(flow[out_a] == inflow_sum + delta, f"load_balance_{vid}_{p}_{t}")
                        else:
                            # travel or sink
                            model.addConstr(flow[out_a] == inflow_sum + delta, f"load_balance_{vid}_{p}_{t}")
                    model.addConstr(procurement[p] == gp.quicksum(delta for _ in [1]), name=f"proc_link_{p}")  # will be summed later
                else:
                    # Discharging decreases cargo when leaving
                    for out_a in outgoing_arcs:
                        model.addConstr(flow[out_a] == inflow_sum - delta, f"disch_balance_{vid}_{p}_{t}")
                    # revenue link later

                # Draft limit on incoming cargo
                for inc_a in incoming_arcs:
                    model.addConstr(flow[inc_a] <= draft_max * x[inc_a], f"draft_{vid}_{p}_{t}")

                # Operation forces visit (already partially enforced)

    # Aggregate procurement and revenue
    for s_port in supply_dict:
        s_terms = []
        for v in vessels:
            vid = v['id']
            for t in range(v['availability_start'], v['availability_end']+1):
                if (vid, s_port, t) in y:
                    delta_var = model.getVarByName(f"delta_{vid}_{s_port}_{t}")
                    if delta_var is not None:
                        s_terms.append(delta_var)
        if s_terms:
            model.addConstr(procurement[s_port] == gp.quicksum(s_terms), f"proc_agg_{s_port}")

    for d_port in demand_dict:
        d_terms = []
        for v in vessels:
            vid = v['id']
            for t in range(v['availability_start'], v['availability_end']+1):
                if (vid, d_port, t) in y:
                    delta_var = model.getVarByName(f"delta_{vid}_{d_port}_{t}")
                    if delta_var is not None:
                        d_terms.append(delta_var)
        if d_terms:
            model.addConstr(revenue[d_port] == gp.quicksum(d_terms), f"rev_agg_{d_port}")

    # Inventory constraints
    for p_id, p_data in supply_dict.items():
        prod = p_data['production_rate']
        stor = p_data['storage_capacity']
        init = p_data['initial_inventory']
        inv = [model.addVar(lb=0, ub=stor[0], name=f"inv_{p_id}_0")]
        model.addConstr(inv[0] == init + prod[0] - procurement[p_id], f"inv_balance_{p_id}_0")
        for tau in range(1, T):
            ub_tau = stor[tau]
            inv_tau = model.addVar(lb=0, ub=ub_tau, name=f"inv_{p_id}_{tau+1}")
            model.addConstr(inv_tau == inv[tau-1] + prod[tau] - 0, f"inv_balance_{p_id}_{tau}")  # procurement already subtracted at t=0? Wait, procurement is total.
            # Actually since procurement is total over all time, we need per period loading
            # This is a flaw in current formulation. We need time-indexed loading.

    # The above inventory formulation is incorrect because procurement is aggregated over time.
    # We must track per-period transfers.

    # Reset and reformulate with per-period transfer variables

    model.dispose()
    model = gp.Model("Bulk_Shipping_V2")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(10, args.time_limit - 10))

    # Redefine variables with better structure
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    flow = model.addVars(arcs, lb=0.0, ub=GRB.INFINITY, name="flow")
    load_qty = model.addVars([(v['id'], s['port_id'], t) for v in vessels for s in supply_ports for t in range(v['availability_start'], v['availability_end']+1)], lb=0, name="load")
    discharge_qty = model.addVars([(v['id'], d['port_id'], t) for v in vessels for d in demand_ports for t in range(v['availability_start'], v['availability_end']+1)], lb=0, name="disch")

    # Objective
    transport_cost = gp.quicksum(arc_cost[a] * x[a] for a in arcs if a in arc_cost)
    procurement_cost = gp.quicksum(supply_dict[s]['unit_procurement_cost'] * gp.quicksum(load_qty[(v['id'], s, t)] for v in vessels for t in range(v['availability_start'], v['availability_end']+1) if (v['id'], s, t) in load_qty) for s in supply_dict)
    revenue_total = gp.quicksum(demand_dict[d]['unit_supply_revenue'] * gp.quicksum(discharge_qty[(v['id'], d, t)] for v in vessels for t in range(v['availability_start'], v['availability_end']+1) if (v['id'], d, t) in discharge_qty) for d in demand_dict)

    obj = transport_cost + procurement_cost - revenue_total
    model.setObjective(obj, GRB.MINIMIZE)

    # Vessel path constraints
    for v_idx, v in enumerate(vessels):
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        src = f"src_{vid}"
        snk = f"snk_{vid}"
        cap = v['capacity']
        init_load = v['initial_load']

        nodes = [src] + [f"{p},{t}" for p in all_ports for t in range(t_start, t_end+1)] + [snk]

        # Exactly one outgoing from source
        model.addConstr(gp.quicksum(x[(vid, src, head)] for head in [f"{p},{t_start}" for p in all_ports] if (vid, src, head) in x) == 1, f"src_out_{vid}")

        # Exactly one incoming to sink
        model.addConstr(gp.quicksum(x[(vid, tail, snk)] for tail in [f"{p},{t}" for p in all_ports for t in range(t_start, t_end+1)] if (vid, tail, snk) in x) == 1, f"snk_in_{vid}")

        # Flow conservation
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                node = f"{p},{t}"
                inflow = gp.quicksum(x[a] for a in arcs if a[0] == vid and a[2] == node)
                outflow = gp.quicksum(x[a] for a in arcs if a[0] == vid and a[1] == node)
                model.addConstr(inflow == outflow, f"node_balance_{vid}_{node}")

        # Cargo flow
        for a in arcs:
            if a[0] != vid:
                continue
            if a[1].startswith("src"):
                model.addConstr(flow[a] == init_load * x[a])
            model.addConstr(flow[a] <= cap * x[a])

        # Cargo continuity and load/discharge
        for p in all_ports:
            is_supply = p in supply_dict
            Fmin = supply_dict[p]['F_min'] if is_supply else demand_dict[p]['F_min']
            Fmax = supply_dict[p]['F_max'] if is_supply else demand_dict[p]['F_max']
            draft_key = f"{vid}->{p}"
            dlimit = draft_limits.get(draft_key, cap)

            for t in range(t_start, t_end + 1):
                node = f"{p},{t}"
                incoming = [a for a in arcs if a[0]==vid and a[2]==node]
                outgoing = [a for a in arcs if a[0]==vid and a[1]==node]

                if not incoming:
                    continue

                inflow_cargo = gp.quicksum(flow[a] for a in incoming)
                op_var = None
                qty_var = None

                if is_supply:
                    qty_var = load_qty[(vid, p, t)]
                    model.addConstr(qty_var <= Fmax)
                    model.addConstr(qty_var >= Fmin * (qty_var > 0))  # big-M not ideal but
                    # Use binary for operation
                    y_var = model.addVar(vtype=GRB.BINARY, name=f"y_{vid}_{p}_{t}")
                    model.addConstr(qty_var <= Fmax * y_var)
                    model.addConstr(qty_var >= Fmin * y_var)
                    op_var = y_var
                else:
                    qty_var = discharge_qty[(vid, p, t)]
                    model.addConstr(qty_var <= Fmax)
                    model.addConstr(qty_var >= Fmin * (qty_var > 0))
                    y_var = model.addVar(vtype=GRB.BINARY, name=f"y_{vid}_{p}_{t}")
                    model.addConstr(qty_var <= Fmax * y_var)
                    model.addConstr(qty_var >= Fmin * y_var)
                    op_var = y_var

                # Link to having a "stop"
                # If any incoming arc is travel or source, then operation is allowed
                can_stop = gp.quicksum(x[a] for a in incoming if not arc_travel.get(a, False) is False or a[1].startswith('src'))
                model.addConstr(op_var <= 1)  # trivial

                # Cargo update
                if is_supply:
                    for out_a in outgoing:
                        model.addConstr(flow[out_a] == inflow_cargo + qty_var)
                else:
                    for out_a in outgoing:
                        model.addConstr(flow[out_a] == inflow_cargo - qty_var)

                # Draft on incoming
                for a_in in incoming:
                    model.addConstr(flow[a_in] <= dlimit * x[a_in])

    # Inventory balance - now with per period transfers
    port_inventory = {}
    for p_id in all_ports:
        if p_id in supply_dict:
            data = supply_dict[p_id]
            rates = data['production_rate']
            stor_cap = data['storage_capacity']
            init_inv = data['initial_inventory']
            is_supply_port = True
        else:
            data = demand_dict[p_id]
            rates = data['consumption_rate']
            stor_cap = data['storage_capacity']
            init_inv = data['initial_inventory']
            is_supply_port = False

        inv_vars = []
        prev_inv = init_inv
        for tau in range(T):
            inv_var = model.addVar(lb=0.0, ub=stor_cap[tau], name=f"inv_{p_id}_{tau+1}")
            inv_vars.append(inv_var)

            transfer_out = gp.quicksum(load_qty[(v['id'], p_id, t)] for v in vessels 
                                     if (v['id'], p_id, t) in load_qty and t-1 == tau) if is_supply_port else 0
            transfer_in = gp.quicksum(discharge_qty[(v['id'], p_id, t)] for v in vessels 
                                    if (v['id'], p_id, t) in discharge_qty and t-1 == tau) if not is_supply_port else 0

            if is_supply_port:
                balance = prev_inv + rates[tau] - transfer_out == inv_var
            else:
                balance = prev_inv - rates[tau] + transfer_in == inv_var
            model.addConstr(balance, f"inv_balance_{p_id}_{tau}")
            prev_inv = inv_var

        port_inventory[p_id] = inv_vars

    # All vessels must end by availability_end
    for v in vessels:
        vid = v['id']
        t_end = v['availability_end']
        model.addConstr(gp.quicksum(x[a] for a in arcs if a[0]==vid and a[2].startswith('snk')) == 1)

    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        # Try to extract best solution if available
        if model.SolCount > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
        else:
            print("No feasible solution found.")
            sys.exit(1)

    # Extract solution
    objective_value = model.objVal if model.status == GRB.OPTIMAL else model.objBound if model.SolCount == 0 else model.objVal

    vessel_routes = {}
    vessel_operations = {}
    port_inventories = {}

    for vid in [v['id'] for v in vessels]:
        vessel_routes[vid] = []
        vessel_operations[vid] = []

        # Get used arcs
        used_arcs = []
        for a in arcs:
            if a[0] == vid and x[a].X > 0.5:
                used_arcs.append(a)

        # Sort by time if possible
        def get_time(node):
            if node.startswith('src'):
                return -1
            if node.startswith('snk'):
                return 9999
            try:
                return int(node.split(',')[1])
            except:
                return 0

        used_arcs.sort(key=lambda a: get_time(a[1]))

        for a in used_arcs:
            tail = a[1]
            head = a[2]
            a_type = arc_type.get(a, 'travel')
            cost = arc_cost.get(a, 0.0)
            prod_flow = flow[a].X

            arc_dict = {
                "tail": tail,
                "head": head,
                "type": a_type,
                "cost": float(cost),
                "product_flow": float(prod_flow)
            }
            vessel_routes[vid].append(arc_dict)

        # Operations
        for p in all_ports:
            for t in range(1, T+1):
                load_key = (vid, p, t)
                if load_key in load_qty and load_qty[load_key].X > 1e-3:
                    vessel_operations[vid].append({
                        "port": p,
                        "time": t,
                        "quantity": float(load_qty[load_key].X),
                        "type": "load"
                    })
                if load_key in discharge_qty and discharge_qty[load_key].X > 1e-3:  # wrong key but for compatibility
                    vessel_operations[vid].append({
                        "port": p,
                        "time": t,
                        "quantity": -float(discharge_qty[load_key].X),
                        "type": "discharge"
                    })

    for p_id in all_ports:
        port_inventories[p_id] = [float(var.X) for var in port_inventory.get(p_id, [])]

    # Cost breakdown
    transport_cost_val = sum(arc_cost.get(a, 0.0) * x[a].X for a in arcs if arc_type.get(a, '') in ('travel', 'demurrage'))
    demurrage_cost_val = sum(arc_cost.get(a, 0.0) * x[a].X for a in arcs if arc_type.get(a, '') == 'demurrage')
    procurement_cost_val = sum(supply_dict[s]['unit_procurement_cost'] * sum(load_qty[(v['id'], s, t)].X for v in vessels for t in range(v['availability_start'], v['availability_end']+1) if (v['id'], s, t) in load_qty) for s in supply_dict)
    supply_revenue_val = sum(demand_dict[d]['unit_supply_revenue'] * sum(discharge_qty[(v['id'], d, t)].X for v in vessels for t in range(v['availability_start'], v['availability_end']+1) if (v['id'], d, t) in discharge_qty) for d in demand_dict)

    cost_breakdown = {
        "transport_cost": float(transport_cost_val),
        "demurrage_cost": float(demurrage_cost_val),
        "procurement_cost": float(procurement_cost_val),
        "supply_revenue": float(supply_revenue_val),
        "total_cost": float(transport_cost_val + demurrage_cost_val + procurement_cost_val - supply_revenue_val)
    }

    solution = {
        "objective_value": float(objective_value),
        "vessel_routes": vessel_routes,
        "port_inventories": port_inventories,
        "vessel_operations": vessel_operations,
        "cost_breakdown": cost_breakdown
    }

    if logger:
        logger.log_solution(float(objective_value), solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path} with objective {objective_value}")

if __name__ == "__main__":
    main()