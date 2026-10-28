import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import sys
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    planning_horizon = instance['planning_horizon']
    vessels = instance['vessels']
    supply_ports = instance['supply_ports']
    demand_ports = instance['demand_ports']
    travel_times = instance['travel_times']
    draft_limits = instance['draft_limits']
    
    # Combine ports and create port info dictionary
    all_ports = [p['port_id'] for p in supply_ports] + [p['port_id'] for p in demand_ports]
    port_info = {}
    for p in supply_ports + demand_ports:
        port_info[p['port_id']] = p
    
    # Create model
    m = gp.Model('vessel_routing')
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', time_limit)
    
    # Data structures for variables and vessel arcs
    x = {}  # (v_id, tail, head) -> binary arc selection
    y = {}  # (v_id, tail, head) -> continuous cargo flow
    f = {}  # (v_id, port, time) -> continuous product transfer
    w = {}  # (v_id, port, time) -> binary operation indicator
    vessel_arcs = {}  # vessel_id -> list of arcs (tail, head)
    
    # Build network per vessel
    for vessel in vessels:
        v_id = vessel['id']
        a_start = vessel['availability_start']
        a_end = min(vessel['availability_end'], planning_horizon)
        vessel_arcs[v_id] = []
        
        # Nodes
        source_node = ('source', v_id)
        sink_node = ('sink', v_id)
        real_nodes = [(port, t) for port in all_ports for t in range(a_start, a_end+1)]
        
        # Arcs
        arcs = []
        # Source arcs
        for port in all_ports:
            arc = (source_node, (port, a_start))
            arcs.append(arc)
            vessel_arcs[v_id].append(arc)
        # Sink arcs
        for port in all_ports:
            for t in range(a_start, a_end+1):
                arc = ((port, t), sink_node)
                arcs.append(arc)
                vessel_arcs[v_id].append(arc)
        # Waiting arcs (same port, consecutive times)
        for port in all_ports:
            for t in range(a_start, a_end):
                arc = ((port, t), (port, t+1))
                arcs.append(arc)
                vessel_arcs[v_id].append(arc)
        # Sailing arcs (different ports)
        for p1 in all_ports:
            for p2 in all_ports:
                if p1 == p2:
                    continue
                key = f"{p1}->{p2}"
                d = travel_times.get(key)
                if d is None:
                    continue
                for t in range(a_start, a_end+1):
                    if t + d <= a_end:
                        arc = ((p1, t), (p2, t+d))
                        arcs.append(arc)
                        vessel_arcs[v_id].append(arc)
        
        # Add arc variables
        for arc in arcs:
            tail, head = arc
            x_key = (v_id, tail, head)
            x[x_key] = m.addVar(vtype=GRB.BINARY, name=f"x_{v_id}_{tail}_{head}")
            y[x_key] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{v_id}_{tail}_{head}")
        
        # Add operation variables and constraints
        for node in real_nodes:
            port, t = node
            f_key = (v_id, port, t)
            f[f_key] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"f_{v_id}_{port}_{t}")
            w_key = (v_id, port, t)
            w[w_key] = m.addVar(vtype=GRB.BINARY, name=f"w_{v_id}_{port}_{t}")
            
            # Operation constraints based on port type
            if port in [p['port_id'] for p in supply_ports]:
                F_min = port_info[port]['F_min']
                F_max = port_info[port]['F_max']
                m.addConstr(f[f_key] >= w[w_key] * F_min)
                m.addConstr(f[f_key] <= w[w_key] * F_max)
                m.addConstr(f[f_key] >= 0)
            else:
                F_min = port_info[port]['F_min']
                F_max = port_info[port]['F_max']
                m.addConstr(f[f_key] <= w[w_key] * (-F_min))
                m.addConstr(f[f_key] >= w[w_key] * (-F_max))
                m.addConstr(f[f_key] <= 0)
            
            # Link operation variable to incoming arcs
            incoming_arcs = [arc for arc in arcs if arc[1] == node]
            w_constr = gp.quicksum(x[(v_id, tail, head)] for tail, head in incoming_arcs)
            m.addConstr(w[w_key] <= w_constr)
        
        # Flow conservation for vessels
        source_out = [arc for arc in arcs if arc[0] == source_node]
        m.addConstr(gp.quicksum(x[(v_id, tail, head)] for tail, head in source_out) == 1)
        
        sink_in = [arc for arc in arcs if arc[1] == sink_node]
        m.addConstr(gp.quicksum(x[(v_id, tail, head)] for tail, head in sink_in) == 1)
        
        for node in real_nodes:
            incoming = [arc for arc in arcs if arc[1] == node]
            outgoing = [arc for arc in arcs if arc[0] == node]
            lhs = gp.quicksum(x[(v_id, tail, head)] for tail, head in incoming)
            rhs = gp.quicksum(x[(v_id, tail, head)] for tail, head in outgoing)
            m.addConstr(lhs == rhs)
        
        # Flow conservation for cargo
        for tail, head in source_out:
            m.addConstr(y[(v_id, tail, head)] == vessel['initial_load'] * x[(v_id, tail, head)])
        
        for node in real_nodes:
            incoming = [arc for arc in arcs if arc[1] == node]
            outgoing = [arc for arc in arcs if arc[0] == node]
            lhs = gp.quicksum(y[(v_id, tail, head)] for tail, head in outgoing)
            rhs = gp.quicksum(y[(v_id, tail, head)] for tail, head in incoming) + f[(v_id, node[0], node[1])]
            m.addConstr(lhs == rhs)
            
            # Draft limit constraint
            draft_key = f"{v_id}->{node[0]}"
            draft = draft_limits.get(draft_key)
            if draft is not None:
                cargo_arrival = gp.quicksum(y[(v_id, tail, head)] for tail, head in incoming)
                m.addConstr(cargo_arrival <= draft)
        
        # Vessel capacity constraints on arcs
        for arc in arcs:
            tail, head = arc
            if tail == source_node or head == sink_node:
                continue
            arc_key = (v_id, tail, head)
            m.addConstr(y[arc_key] <= vessel['capacity'] * x[arc_key])
    
    # Port inventory constraints
    total_f = {}
    for port in all_ports:
        for t in range(planning_horizon + 1):
            total_f[(port, t)] = gp.LinExpr()
    
    for vessel in vessels:
        v_id = vessel['id']
        a_start = vessel['availability_start']
        a_end = min(vessel['availability_end'], planning_horizon)
        for port in all_ports:
            for t in range(a_start, a_end + 1):
                f_key = (v_id, port, t)
                if f_key in f:
                    total_f[(port, t)] += f[f_key]
    
    for port in all_ports:
        if port in [p['port_id'] for p in supply_ports]:
            I_prev = port_info[port]['initial_inventory']
            prod = port_info[port]['production_rate']
            cap = port_info[port]['storage_capacity']
            for t in range(1, planning_horizon + 1):
                I_t = I_prev + prod[t-1] - total_f[(port, t)]
                m.addConstr(I_t >= 0)
                m.addConstr(I_t <= cap[t-1])
                I_prev = I_t
        else:
            I_prev = port_info[port]['initial_inventory']
            cons = port_info[port]['consumption_rate']
            cap = port_info[port]['storage_capacity']
            for t in range(1, planning_horizon + 1):
                I_t = I_prev - cons[t-1] - total_f[(port, t)]
                m.addConstr(I_t >= 0)
                m.addConstr(I_t <= cap[t-1])
                I_prev = I_t
    
    # Objective function
    transport_cost = gp.LinExpr()
    demurrage_cost = gp.LinExpr()
    procurement_cost = gp.LinExpr()
    supply_revenue = gp.LinExpr()
    
    for vessel in vessels:
        v_id = vessel['id']
        a_start = vessel['availability_start']
        a_end = min(vessel['availability_end'], planning_horizon)
        
        # Transport and demurrage costs
        for arc in vessel_arcs[v_id]:
            tail, head = arc
            arc_key = (v_id, tail, head)
            if tail == ('source', v_id) or head == ('sink', v_id):
                continue
            if tail[0] == head[0]:  # Waiting arc
                demurrage_cost += vessel['demurrage_cost_per_day'] * x[arc_key]
            else:  # Sailing arc
                d = head[1] - tail[1]
                transport_cost += d * vessel['transport_cost_per_day'] * x[arc_key]
        
        # Procurement and revenue costs
        for port in all_ports:
            for t in range(a_start, a_end + 1):
                f_key = (v_id, port, t)
                if f_key in f:
                    if port in [p['port_id'] for p in supply_ports]:
                        procurement_cost += f[f_key] * port_info[port]['unit_procurement_cost']
                    else:
                        supply_revenue += (-f[f_key]) * port_info[port]['unit_supply_revenue']
    
    total_cost = transport_cost + demurrage_cost + procurement_cost - supply_revenue
    m.setObjective(total_cost, GRB.MINIMIZE)
    
    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
    
    # Optimize model
    m.optimize(callback)
    
    # Prepare solution dictionary
    solution = {
        "objective_value": 0.0,
        "vessel_routes": {},
        "port_inventories": {},
        "vessel_operations": {},
        "cost_breakdown": {
            "transport_cost": 0.0,
            "demurrage_cost": 0.0,
            "procurement_cost": 0.0,
            "supply_revenue": 0.0,
            "total_cost": 0.0
        }
    }
    
    # Extract solution if available
    if m.SolCount > 0:
        # Helper function to format nodes
        def format_node(node, v_id):
            if node == ('source', v_id):
                return f"('source', '{v_id}')"
            elif node == ('sink', v_id):
                return f"('sink', '{v_id}')"
            else:
                port, time = node
                return f"('{port}', {time})"
        
        # Initialize cost breakdown
        transport_cost_val = 0
        demurrage_cost_val = 0
        procurement_cost_val = 0
        supply_revenue_val = 0
        
        # Extract vessel routes and operations
        for vessel in vessels:
            v_id = vessel['id']
            solution["vessel_routes"][v_id] = []
            solution["vessel_operations"][v_id] = []
            
            # Process arcs
            for arc in vessel_arcs[v_id]:
                tail, head = arc
                arc_key = (v_id, tail, head)
                if x[arc_key].X > 0.5:
                    # Determine arc type
                    if tail == ('source', v_id):
                        arc_type = 'source'
                        cost_val = 0.0
                    elif head == ('sink', v_id):
                        arc_type = 'sink'
                        cost_val = 0.0
                    else:
                        if tail[0] == head[0]:
                            arc_type = 'demurrage'
                            cost_val = vessel['demurrage_cost_per_day']
                            demurrage_cost_val += cost_val
                        else:
                            arc_type = 'travel'
                            d = head[1] - tail[1]
                            cost_val = d * vessel['transport_cost_per_day']
                            transport_cost_val += cost_val
                    
                    # Format nodes
                    tail_str = format_node(tail, v_id)
                    head_str = format_node(head, v_id)
                    
                    # Add to routes
                    solution["vessel_routes"][v_id].append({
                        "tail": tail_str,
                        "head": head_str,
                        "type": arc_type,
                        "cost": cost_val,
                        "product_flow": y[arc_key].X
                    })
            
            # Process operations
            a_start = vessel['availability_start']
            a_end = min(vessel['availability_end'], planning_horizon)
            for port in all_ports:
                for t in range(a_start, a_end + 1):
                    f_key = (v_id, port, t)
                    if f_key in f and abs(f[f_key].X) > 1e-6:
                        if port in [p['port_id'] for p in supply_ports]:
                            op_type = 'load'
                            procurement_cost_val += f[f_key].X * port_info[port]['unit_procurement_cost']
                        else:
                            op_type = 'discharge'
                            supply_revenue_val += (-f[f_key].X) * port_info[port]['unit_supply_revenue']
                        
                        solution["vessel_operations"][v_id].append({
                            "port": port,
                            "time": t,
                            "quantity": f[f_key].X,
                            "type": op_type
                        })
        
        # Compute port inventories
        for port in all_ports:
            inventory = [0] * planning_horizon
            I_prev = port_info[port]['initial_inventory']
            if port in [p['port_id'] for p in supply_ports]:
                prod = port_info[port]['production_rate']
                for t in range(1, planning_horizon + 1):
                    total_f_val = 0
                    for vessel in vessels:
                        v_id = vessel['id']
                        f_key = (v_id, port, t)
                        if f_key in f:
                            total_f_val += f[f_key].X
                    I_t = I_prev + prod[t-1] - total_f_val
                    inventory[t-1] = I_t
                    I_prev = I_t
            else:
                cons = port_info[port]['consumption_rate']
                for t in range(1, planning_horizon + 1):
                    total_f_val = 0
                    for vessel in vessels:
                        v_id = vessel['id']
                        f_key = (v_id, port, t)
                        if f_key in f:
                            total_f_val += f[f_key].X
                    I_t = I_prev - cons[t-1] - total_f_val
                    inventory[t-1] = I_t
                    I_prev = I_t
            solution["port_inventories"][port] = inventory
        
        # Set cost breakdown
        total_cost_val = transport_cost_val + demurrage_cost_val + procurement_cost_val - supply_revenue_val
        solution["objective_value"] = total_cost_val
        solution["cost_breakdown"]["transport_cost"] = transport_cost_val
        solution["cost_breakdown"]["demurrage_cost"] = demurrage_cost_val
        solution["cost_breakdown"]["procurement_cost"] = procurement_cost_val
        solution["cost_breakdown"]["supply_revenue"] = supply_revenue_val
        solution["cost_breakdown"]["total_cost"] = total_cost_val
        
        # Log final solution
        if logger:
            logger.log_solution(total_cost_val, solution)
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)