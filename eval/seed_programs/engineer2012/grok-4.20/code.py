import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
from collections import defaultdict

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

    vessel_list = [v['id'] for v in vessels]
    supply_list = [p['port_id'] for p in supply_ports]
    demand_list = [p['port_id'] for p in demand_ports]
    all_ports = supply_list + demand_list

    supply_data = {p['port_id']: p for p in supply_ports}
    demand_data = {p['port_id']: p for p in demand_ports}
    vessel_data = {v['id']: v for v in vessels}

    # Precompute all valid arcs: (vid, tail_node, head_node)
    arcs = []
    arc_dict = {}  # (vid, tail, head) -> info

    for v in vessels:
        vid = v['id']
        t_start = v['availability_start']
        t_end = v['availability_end']
        src = f"src_{vid}"
        snk = f"snk_{vid}"
        cap = float(v['capacity'])
        dem_cost = float(v['demurrage_cost_per_day'])
        trans_cost = float(v['transport_cost_per_day'])
        init_load = float(v.get('initial_load', 0))

        # Source arcs
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                head = f"{p},{t}"
                a = (vid, src, head)
                arcs.append(a)
                arc_dict[a] = {
                    'type': 'source',
                    'cost': 0.0,
                    'travel': False,
                    'tt': 0,
                    'port': p,
                    'time': t,
                    'from_src': True
                }

        # Sink arcs
        for p in all_ports:
            for t in range(t_start, t_end + 1):
                tail = f"{p},{t}"
                a = (vid, tail, snk)
                arcs.append(a)
                arc_dict[a] = {
                    'type': 'sink',
                    'cost': 0.0,
                    'travel': False,
                    'tt': 0,
                    'port': p,
                    'time': t,
                    'from_src': False
                }

        # Demurrage and travel arcs
        for p in all_ports:
            for t in range(t_start, t_end):
                tail_node = f"{p},{t}"
                # Demurrage
                head_node = f"{p},{t+1}"
                a = (vid, tail_node, head_node)
                arcs.append(a)
                arc_dict[a] = {
                    'type': 'demurrage',
                    'cost': dem_cost,
                    'travel': False,
                    'tt': 1,
                    'port': p,
                    'time': t + 1,
                    'from_src': False
                }

                # Travel to other ports
                for q in all_ports:
                    if p == q:
                        continue
                    key = f"{p}->{q}"
                    if key not in travel_times:
                        continue
                    tt = int(travel_times[key])
                    arrival = t + tt
                    if arrival > t_end:
                        continue
                    head_node = f"{q},{arrival}"
                    a = (vid, tail_node, head_node)
                    arcs.append(a)
                    arc_dict[a] = {
                        'type': 'travel',
                        'cost': trans_cost * tt,
                        'travel': True,
                        'tt': tt,
                        'port': q,
                        'time': arrival,
                        'from_src': False
                    }

    model = gp.Model("BulkShipping")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(10, args.time_limit - 10))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    flow = model.addVars(arcs, lb=0.0, ub=GRB.INFINITY, name="flow")

    # Transfer amount at each (vessel, port, time). Positive = load, negative = discharge
    transfer = {}
    for vid in vessel_list:
        v = vessel_data[vid]
        t_start = v['availability_start']
        t_end = v['availability_end']
        for p_id in all_ports:
            for t in range(t_start, t_end + 1):
                transfer[(vid, p_id, t)] = model.addVar(
                    lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"trans_{vid}_{p_id}_{t}"
                )

    # Objective components
    voyage_cost = gp.quicksum(arc_dict[a]['cost'] * x[a] for a in arcs)

    procurement_cost = gp.LinExpr()
    for p in supply_list:
        c = supply_data[p]['unit_procurement_cost']
        for vid in vessel_list:
            for t in range(1, T+1):
                key = (vid, p, t)
                if key in transfer:
                    procurement_cost.add(transfer[key] * c)

    revenue_expr = gp.LinExpr()
    for p in demand_list:
        r = demand_data[p]['unit_supply_revenue']
        for vid in vessel_list:
            for t in range(1, T+1):
                key = (vid, p, t)
                if key in transfer:
                    revenue_expr.add(transfer[key] * (-r))

    model.setObjective(voyage_cost + procurement_cost + revenue_expr, GRB.MINIMIZE)

    # Flow conservation for each vessel
    for vid in vessel_list:
        v = vessel_data[vid]
        t_start = v['availability_start']
        t_end = v['availability_end']
        src = f"src_{vid}"
        snk = f"snk_{vid}"

        model.addConstr(gp.quicksum(x[a] for a in arcs if a[0]==vid and a[1]==src) == 1, f"src_{vid}")
        model.addConstr(gp.quicksum(x[a] for a in arcs if a[0]==vid and a[2]==snk) == 1, f"snk_{vid}")

        for p in all_ports:
            for t in range(t_start, t_end + 1):
                node = f"{p},{t}"
                inflow = gp.quicksum(x[a] for a in arcs if a[0]==vid and a[2]==node)
                outflow = gp.quicksum(x[a] for a in arcs if a[0]==vid and a[1]==node)
                model.addConstr(inflow - outflow == 0, f"bal_{vid}_{p}_{t}")

    # Cargo flow and transfer linking + activity detection
    M = 1e7
    for vid in vessel_list:
        v = vessel_data[vid]
        t_start = v['availability_start']
        t_end = v['availability_end']
        cap = float(v['capacity'])
        init_load = float(v.get('initial_load', 0.0))

        for a_idx, a in enumerate(arcs):
            if a[0] != vid: continue
            info = arc_dict[a]
            if info['from_src']:
                model.addConstr(flow[a] == init_load * x[a], f"initflow_{vid}_{a_idx}")
            model.addConstr(flow[a] <= cap * x[a], f"capflow_{a_idx}")

        for p_id in all_ports:
            is_supply = p_id in supply_list
            pdata = supply_data[p_id] if is_supply else demand_data[p_id]
            Fmin = float(pdata['F_min'])
            Fmax = float(pdata['F_max'])
            draft_key = f"{vid}->{p_id}"
            draft_max = float(draft_limits.get(draft_key, cap))

            for t in range(t_start, t_end + 1):
                node = f"{p_id},{t}"
                in_arcs = [a for a in arcs if a[0] == vid and a[2] == node]
                out_arcs = [a for a in arcs if a[0] == vid and a[1] == node]

                if not in_arcs or not out_arcs:
                    continue

                y = model.addVar(vtype=GRB.BINARY, name=f"yop_{vid}_{p_id}_{t}")
                q = transfer[(vid, p_id, t)]

                model.addConstr(q <= Fmax * y, f"qmax_{vid}_{p_id}_{t}")
                model.addConstr(q >= -Fmax * y, f"qmin1_{vid}_{p_id}_{t}")
                model.addConstr(q <= M * y, f"qmin2_{vid}_{p_id}_{t}")
                model.addConstr(q >= -M * y, f"qmin3_{vid}_{p_id}_{t}")

                inflow_cargo = gp.quicksum(flow[a] for a in in_arcs)
                for oa in out_arcs:
                    if is_supply:
                        model.addConstr(flow[oa] == inflow_cargo + q, f"flowlink_s_{vid}_{p_id}_{t}")
                    else:
                        model.addConstr(flow[oa] == inflow_cargo - q, f"flowlink_d_{vid}_{p_id}_{t}")

                for ia in in_arcs:
                    model.addConstr(flow[ia] <= draft_max * x[ia], f"draft_{vid}_{p_id}_{t}")

    # Inventory balance
    for p_id in all_ports:
        if p_id in supply_data:
            pdata = supply_data[p_id]
            rates = pdata['production_rate']
            capacities = pdata['storage_capacity']
            init = float(pdata['initial_inventory'])
            sign = 1.0
        else:
            pdata = demand_data[p_id]
            rates = pdata['consumption_rate']
            capacities = pdata['storage_capacity']
            init = float(pdata['initial_inventory'])
            sign = -1.0

        prev = init
        for tau in range(T):
            t = tau + 1
            total_q = gp.quicksum(transfer.get((v, p_id, t), 0.0) for v in vessel_list)
            inv = model.addVar(lb=0.0, ub=float(capacities[tau]), name=f"inv_{p_id}_{t}")
            model.addConstr(prev + rates[tau] - sign * total_q == inv, f"invbal_{p_id}_{t}")
            prev = inv

    # Callback for logging
    model._best_obj = float('inf')
    model._logger = logger
    model._start_time = start_time
    model._time_limit = args.time_limit

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < model._best_obj - 1e-6:
                model._best_obj = obj
                if model._logger:
                    try:
                        model._logger.log(obj)
                    except:
                        pass

    model.optimize(callback)

    if model.SolCount == 0:
        print("ERROR: No feasible solution found within time limit.")
        sys.exit(1)

    # ====================== EXTRACT SOLUTION ======================
    sol = {
        "objective_value": float(model.objVal),
        "vessel_routes": {},
        "port_inventories": {},
        "vessel_operations": {},
        "cost_breakdown": {}
    }

    # Vessel routes
    for vid in vessel_list:
        route_list = []
        used_arcs = [a for a in arcs if a[0] == vid and abs(x[a].X) > 0.5]
        used_arcs.sort(key=lambda a: arc_dict[a].get('time', 0))

        for a in used_arcs:
            info = arc_dict[a]
            route_list.append({
                "tail": str(a[1]),
                "head": str(a[2]),
                "type": info['type'],
                "cost": float(info.get('cost', 0.0)),
                "product_flow": float(flow[a].X)
            })

        sol["vessel_routes"][vid] = route_list

    # Vessel operations
    for vid in vessel_list:
        ops = []
        v = vessel_data[vid]
        t_start = v['availability_start']
        t_end = v['availability_end']
        for p in all_ports:
            for t in range(max(t_start, 1), min(t_end, T) + 1):
                qvar = transfer.get((vid, p, t))
                if qvar is None:
                    continue
                qty = qvar.X
                if abs(qty) > 1e-4:
                    op_type = "load" if qty > 0 else "discharge"
                    ops.append({
                        "port": p,
                        "time": int(t),
                        "quantity": float(qty),
                        "type": op_type
                    })
        sol["vessel_operations"][vid] = ops

    # Port inventories (periods 1 to T)
    for p in all_ports:
        inv_list = []
        for t in range(1, T + 1):
            var_name = f"inv_{p}_{t}"
            var = model.getVarByName(var_name)
            val = float(var.X) if var is not None else 0.0
            inv_list.append(val)
        sol["port_inventories"][p] = inv_list

    # Cost breakdown
    transport = sum(arc_dict[a]['cost'] * x[a].X for a in arcs if arc_dict[a].get('type') == 'travel')
    demurrage = sum(arc_dict[a]['cost'] * x[a].X for a in arcs if arc_dict[a].get('type') == 'demurrage')
    proc = 0.0
    for p in supply_list:
        c = supply_data[p]['unit_procurement_cost']
        for vid in vessel_list:
            for t in range(1, T+1):
                qv = transfer.get((vid, p, t))
                if qv:
                    proc += qv.X * c
    rev = 0.0
    for p in demand_list:
        r = demand_data[p]['unit_supply_revenue']
        for vid in vessel_list:
            for t in range(1, T+1):
                qv = transfer.get((vid, p, t))
                if qv:
                    rev += (-qv.X) * r

    total_cost = transport + demurrage + proc - rev

    sol["cost_breakdown"] = {
        "transport_cost": float(transport),
        "demurrage_cost": float(demurrage),
        "procurement_cost": float(proc),
        "supply_revenue": float(rev),
        "total_cost": float(total_cost)
    }
    sol["objective_value"] = float(total_cost)

    if logger:
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Final objective: {sol['objective_value']:.6f}")


if __name__ == "__main__":
    main()