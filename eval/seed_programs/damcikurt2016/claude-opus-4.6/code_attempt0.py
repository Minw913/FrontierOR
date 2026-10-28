import argparse
import json
import time
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    gen = data['generator']
    tree = data['scenario_tree']

    # Generator parameters
    C_lower = gen['C_lower_MW']
    C_upper = gen['C_upper_MW']
    V_plus = gen['V_plus_MW_per_h']
    V_minus = gen['V_minus_MW_per_h']
    startup_cost = gen['U_bar_startup_cost_dollars']
    shutdown_cost = gen['U_lower_shutdown_cost_dollars']
    fuel_a = gen['fuel_a_dollars_per_h']
    fuel_b = gen['fuel_b_dollars_per_MWh']

    # Tree parameters
    T = tree['T_num_periods']
    L = tree['L_min_up_time']
    ell = tree['ell_min_down_time']
    y0 = tree['initial_generator_status_y0']
    x0 = tree['initial_generation_x0_MW']
    nodes = tree['nodes']
    total_nodes = tree['total_nodes']

    # Build node lookup
    node_by_id = {}
    for n in nodes:
        node_by_id[n['id']] = n

    # Sort nodes by id for consistent indexing
    node_ids = sorted(node_by_id.keys())

    # Build index mapping
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Identify root
    root_id = None
    for n in nodes:
        if n['parent_id'] is None:
            root_id = n['id']
            break

    # For min-up and min-down constraints, we need ancestor paths
    # For each node, find ancestors within certain time windows
    # Build parent mapping and children mapping
    parent_map = {}
    children_map = {}
    time_map = {}
    price_map = {}
    prob_map = {}

    for n in nodes:
        nid = n['id']
        parent_map[nid] = n['parent_id']
        children_map[nid] = n['children_ids']
        time_map[nid] = n['time']
        price_map[nid] = n['electricity_price_dollars_per_MWh']
        prob_map[nid] = n['probability']

    # For min-up/min-down constraints:
    # If startup at node n (time t), then y must be 1 at all descendants reachable within
    # L-1 periods from n (i.e., times t+1, ..., t+L-1)
    # If shutdown at node n (time t), then y must be 0 at all descendants reachable within
    # ell-1 periods from n (i.e., times t+1, ..., t+ell-1)

    # We need to find, for each node, all descendants within a certain time window
    def get_descendants_within(nid, max_periods):
        """Get all descendants of nid within max_periods periods (including nid itself at period 0)"""
        if max_periods <= 0:
            return []
        result = []
        frontier = children_map[nid]
        t_start = time_map[nid]
        for depth in range(1, max_periods + 1):
            next_frontier = []
            for cid in frontier:
                if time_map[cid] <= t_start + max_periods:
                    result.append(cid)
                    next_frontier.extend(children_map[cid])
            frontier = next_frontier
        return result

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("unit_commitment")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    remaining_time = args.time_limit - (time.time() - start_time)
    model.setParam('TimeLimit', max(remaining_time - 5, 10))

    # Decision variables
    # x[nid] = generation level at node nid
    # y[nid] = on/off status at node nid (binary)
    # u[nid] = startup indicator at non-root node nid (binary)
    # v[nid] = shutdown indicator at non-root node nid (binary, auxiliary)

    x = {}
    y = {}
    u = {}
    v = {}

    for nid in node_ids:
        x[nid] = model.addVar(lb=0, ub=C_upper, vtype=GRB.CONTINUOUS, name=f"x_{nid}")
        y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")

    non_root_ids = [nid for nid in node_ids if nid != root_id]
    for nid in non_root_ids:
        u[nid] = model.addVar(vtype=GRB.BINARY, name=f"u_{nid}")
        v[nid] = model.addVar(vtype=GRB.BINARY, name=f"v_{nid}")

    model.update()

    # Constraints

    # 1. Startup/shutdown logic for non-root nodes
    # v[nid] = y[parent] - y[nid] + u[nid]
    # This means: v[nid] >= 0, and the relationship holds
    for nid in non_root_ids:
        pid = parent_map[nid]
        model.addConstr(v[nid] == y[pid] - y[nid] + u[nid], name=f"sv_logic_{nid}")
        # u[nid] <= y[nid] (startup only if on)
        model.addConstr(u[nid] <= y[nid], name=f"startup_on_{nid}")
        # u[nid] <= 1 - y[parent] (startup only if parent was off)
        model.addConstr(u[nid] <= 1 - y[pid], name=f"startup_off_{nid}")

    # For root node: handle initial conditions
    # If y0=0 and y[root]=1, then startup at root
    # If y0=1 and y[root]=0, then shutdown at root
    # We need u and v for root too for startup/shutdown costs
    u_root = model.addVar(vtype=GRB.BINARY, name=f"u_root_{root_id}")
    v_root = model.addVar(vtype=GRB.BINARY, name=f"v_root_{root_id}")
    model.addConstr(v_root == y0 - y[root_id] + u_root, name=f"sv_logic_root")
    model.addConstr(u_root <= y[root_id], name=f"startup_on_root")
    model.addConstr(u_root <= 1 - y0, name=f"startup_off_root")

    # 2. Generation limits
    for nid in node_ids:
        model.addConstr(x[nid] >= C_lower * y[nid], name=f"gen_lower_{nid}")
        model.addConstr(x[nid] <= C_upper * y[nid], name=f"gen_upper_{nid}")

    # 3. Ramp constraints
    # For non-root nodes:
    # x[nid] - x[parent] <= V_plus * y[parent] + C_upper * (1 - y[parent])
    #   (if parent was on, ramp-up limit; if parent was off, free jump)
    # x[parent] - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid])
    #   (if current is on, ramp-down limit; if current is off, free drop)
    for nid in non_root_ids:
        pid = parent_map[nid]
        # Ramp up: x[nid] - x[pid] <= V_plus if parent on; free if parent off
        model.addConstr(x[nid] - x[pid] <= V_plus * y[pid] + C_upper * (1 - y[pid]),
                        name=f"ramp_up_{nid}")
        # Ramp down: x[pid] - x[nid] <= V_minus if current on; free if current off
        model.addConstr(x[pid] - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid]),
                        name=f"ramp_down_{nid}")

    # Root ramp constraints with initial conditions
    if y0 == 1:
        model.addConstr(x[root_id] - x0 <= V_plus * y0 + C_upper * (1 - y0),
                        name=f"ramp_up_root")
        model.addConstr(x0 - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]),
                        name=f"ramp_down_root")
    else:
        # Parent was off, so x0 = 0 (or whatever it is, but free jump allowed)
        model.addConstr(x[root_id] - x0 <= V_plus * y0 + C_upper * (1 - y0),
                        name=f"ramp_up_root")
        model.addConstr(x0 - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]),
                        name=f"ramp_down_root")

    # 4. Minimum up-time constraints
    # If startup at node nid (time t), y must be 1 for all descendants within L-1 periods
    if L > 1:
        # For root startup
        descendants_root = get_descendants_within(root_id, L - 1)
        for did in descendants_root:
            model.addConstr(y[did] >= u_root, name=f"minup_root_{did}")

        for nid in non_root_ids:
            if L > 1:
                descendants = get_descendants_within(nid, L - 1)
                for did in descendants:
                    model.addConstr(y[did] >= u[nid], name=f"minup_{nid}_{did}")

    # 5. Minimum down-time constraints
    # If shutdown at node nid (time t), y must be 0 for all descendants within ell-1 periods
    if ell > 1:
        # For root shutdown
        descendants_root = get_descendants_within(root_id, ell - 1)
        for did in descendants_root:
            model.addConstr(y[did] <= 1 - v_root, name=f"mindn_root_{did}")

        for nid in non_root_ids:
            if ell > 1:
                descendants = get_descendants_within(nid, ell - 1)
                for did in descendants:
                    model.addConstr(y[did] <= 1 - v[nid], name=f"mindn_{nid}_{did}")

    # Also handle initial conditions for min-up/min-down
    # If initially on (y0=1), check if we need to enforce remaining min-up from before horizon
    # The problem doesn't specify initial run length, so we assume no constraints from before horizon
    # (standard assumption unless stated otherwise)

    # Objective: maximize expected profit
    # profit at each node = prob * (price * x - fuel_cost - startup_cost * u - shutdown_cost * v)
    # fuel_cost = fuel_a * y + fuel_b * x (linear fuel cost)

    obj = gp.LinExpr()

    # Root node contribution
    p_root = prob_map[root_id]
    price_root = price_map[root_id]
    obj += p_root * (price_root * x[root_id] - fuel_a * y[root_id] - fuel_b * x[root_id]
                     - startup_cost * u_root - shutdown_cost * v_root)

    # Non-root nodes
    for nid in non_root_ids:
        p = prob_map[nid]
        price = price_map[nid]
        obj += p * (price * x[nid] - fuel_a * y[nid] - fuel_b * x[nid]
                    - startup_cost * u[nid] - shutdown_cost * v[nid])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Callback for logging
    best_obj = [float('-inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_obj[0] + 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    sol_x = {}
    sol_y = {}
    sol_u = {}

    if model.SolCount > 0:
        obj_val = model.ObjVal

        for nid in node_ids:
            xv = x[nid].X
            yv = round(y[nid].X)
            if xv != 0:
                sol_x[str(nid)] = xv
            if yv != 0:
                sol_y[str(nid)] = yv

        # Root startup
        uv_root = round(u_root.X)
        if uv_root != 0:
            sol_u[str(root_id)] = uv_root

        for nid in non_root_ids:
            uv = round(u[nid].X)
            if uv != 0:
                sol_u[str(nid)] = uv
    else:
        obj_val = 0.0

    solution = {
        "objective_value": obj_val,
        "x": sol_x,
        "y": sol_y,
        "u": sol_u,
        "sparse_defaults": {"x": 0, "y": 0, "u": 0}
    }

    if logger:
        logger.log_solution(obj_val, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()