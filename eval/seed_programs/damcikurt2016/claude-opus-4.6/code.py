import argparse
import json
import time
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    from solution_logger import SolutionLogger
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

    node_ids = sorted(node_by_id.keys())

    root_id = None
    for n in nodes:
        if n['parent_id'] is None:
            root_id = n['id']
            break

    parent_map = {}
    children_map = {}
    time_map = {}
    price_map = {}
    prob_map = {}

    for n in nodes:
        nid = n['id']
        parent_map[nid] = n['parent_id']
        children_map[nid] = n['children_ids'] if n['children_ids'] else []
        time_map[nid] = n['time']
        price_map[nid] = n['electricity_price_dollars_per_MWh']
        prob_map[nid] = n['probability']

    non_root_ids = [nid for nid in node_ids if nid != root_id]

    # Precompute descendants within a given number of periods using BFS
    def get_descendants_within(nid, max_periods):
        if max_periods <= 0:
            return []
        result = []
        t_start = time_map[nid]
        frontier = list(children_map[nid])
        while frontier:
            next_frontier = []
            for cid in frontier:
                if time_map[cid] - t_start <= max_periods:
                    result.append(cid)
                    next_frontier.extend(children_map[cid])
            frontier = next_frontier
        return result

    import gurobipy as gp
    from gurobipy import GRB

    # Create environment explicitly to avoid bus connection issues
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Seed', 0)
    env.setParam('MIPGap', 1e-4)
    env.setParam('NumericFocus', 0)
    env.setParam('Threads', 1)
    env.start()

    model = gp.Model("unit_commitment", env=env)

    remaining_time = args.time_limit - (time.time() - start_time)
    model.setParam('TimeLimit', max(remaining_time - 5, 10))

    # Decision variables
    x = {}
    y = {}
    u = {}
    v = {}

    for nid in node_ids:
        x[nid] = model.addVar(lb=0, ub=C_upper, vtype=GRB.CONTINUOUS, name=f"x_{nid}")
        y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")

    for nid in non_root_ids:
        u[nid] = model.addVar(vtype=GRB.BINARY, name=f"u_{nid}")
        v[nid] = model.addVar(vtype=GRB.BINARY, name=f"v_{nid}")

    # Root startup/shutdown variables
    u_root = model.addVar(vtype=GRB.BINARY, name=f"u_root")
    v_root = model.addVar(vtype=GRB.BINARY, name=f"v_root")

    model.update()

    # Root startup/shutdown logic
    model.addConstr(v_root == y0 - y[root_id] + u_root, name=f"sv_logic_root")
    model.addConstr(u_root <= y[root_id], name=f"startup_on_root")
    model.addConstr(u_root <= 1 - y0, name=f"startup_off_root")

    # Non-root startup/shutdown logic
    for nid in non_root_ids:
        pid = parent_map[nid]
        model.addConstr(v[nid] == y[pid] - y[nid] + u[nid], name=f"sv_{nid}")
        model.addConstr(u[nid] <= y[nid], name=f"su_on_{nid}")
        model.addConstr(u[nid] <= 1 - y[pid], name=f"su_off_{nid}")

    # Generation limits
    for nid in node_ids:
        model.addConstr(x[nid] >= C_lower * y[nid], name=f"gl_{nid}")
        model.addConstr(x[nid] <= C_upper * y[nid], name=f"gu_{nid}")

    # Ramp constraints - root
    model.addConstr(x[root_id] - x0 <= V_plus * y0 + C_upper * (1 - y0), name=f"ru_root")
    model.addConstr(x0 - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]), name=f"rd_root")

    # Ramp constraints - non-root
    for nid in non_root_ids:
        pid = parent_map[nid]
        model.addConstr(x[nid] - x[pid] <= V_plus * y[pid] + C_upper * (1 - y[pid]), name=f"ru_{nid}")
        model.addConstr(x[pid] - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid]), name=f"rd_{nid}")

    # Minimum up-time constraints
    if L > 1:
        descendants_root = get_descendants_within(root_id, L - 1)
        for did in descendants_root:
            model.addConstr(y[did] >= u_root, name=f"mu_r_{did}")

        for nid in non_root_ids:
            descendants = get_descendants_within(nid, L - 1)
            for did in descendants:
                model.addConstr(y[did] >= u[nid], name=f"mu_{nid}_{did}")

    # Minimum down-time constraints
    if ell > 1:
        descendants_root = get_descendants_within(root_id, ell - 1)
        for did in descendants_root:
            model.addConstr(y[did] <= 1 - v_root, name=f"md_r_{did}")

        for nid in non_root_ids:
            descendants = get_descendants_within(nid, ell - 1)
            for did in descendants:
                model.addConstr(y[did] <= 1 - v[nid], name=f"md_{nid}_{did}")

    # Objective: maximize expected profit
    obj = gp.LinExpr()

    p_root = prob_map[root_id]
    price_root = price_map[root_id]
    obj += p_root * ((price_root - fuel_b) * x[root_id] - fuel_a * y[root_id]
                     - startup_cost * u_root - shutdown_cost * v_root)

    for nid in non_root_ids:
        p = prob_map[nid]
        price = price_map[nid]
        obj += p * ((price - fuel_b) * x[nid] - fuel_a * y[nid]
                    - startup_cost * u[nid] - shutdown_cost * v[nid])

    model.setObjective(obj, GRB.MAXIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
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
            if abs(xv) > 1e-9:
                sol_x[str(nid)] = xv
            if yv != 0:
                sol_y[str(nid)] = yv

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

    model.dispose()
    env.dispose()

if __name__ == '__main__':
    main()