import json
import argparse
import sys
from collections import deque

def main():
    parser = argparse.ArgumentParser(description="Unit Commitment on Scenario Tree")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    gen = instance['generator']
    tree = instance['scenario_tree']

    nodes = tree['nodes']
    total_nodes = tree['total_nodes']
    y0 = tree['initial_generator_status_y0']
    x0 = tree['initial_generation_x0_MW']
    L_up = tree['L_min_up_time']
    L_down = tree['ell_min_down_time']

    C_lower = gen['C_lower_MW']
    C_upper = gen['C_upper_MW']
    V_plus = gen['V_plus_MW_per_h']
    V_minus = gen['V_minus_MW_per_h']
    SU_cost = gen['U_bar_startup_cost_dollars']
    SD_cost = gen['U_lower_shutdown_cost_dollars']
    fuel_a = gen['fuel_a_dollars_per_h']
    fuel_b = gen['fuel_b_dollars_per_MWh']

    # Build node lookup
    node_map = {n['id']: n for n in nodes}
    root_id = nodes[0]['id']

    # BFS to find all descendants within k steps
    def get_descendants_within_k(node_id, k):
        if k <= 0:
            return []
        result = []
        queue = deque([(node_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth > 0:
                result.append(current)
            if depth < k:
                for child_id in node_map[current]['children_ids']:
                    queue.append((child_id, depth + 1))
        return result

    # Precompute descendants for min up/down time constraints
    desc_up = {}
    desc_down = {}

    if L_up > 1:
        for n in nodes:
            desc_up[n['id']] = get_descendants_within_k(n['id'], L_up - 1)

    if L_down > 1:
        for n in nodes:
            desc_down[n['id']] = get_descendants_within_k(n['id'], L_down - 1)

    # Formulate MIP with Gurobi
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("unit_commitment")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    y = {}  # on/off status (binary)
    x = {}  # generation level (continuous)
    u = {}  # startup indicator (binary)
    v = {}  # shutdown indicator (binary)

    for n in nodes:
        nid = n['id']
        y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")
        x[nid] = model.addVar(lb=0, ub=C_upper, vtype=GRB.CONTINUOUS, name=f"x_{nid}")
        u[nid] = model.addVar(vtype=GRB.BINARY, name=f"u_{nid}")
        v[nid] = model.addVar(vtype=GRB.BINARY, name=f"v_{nid}")

    model.update()

    # Objective: maximize expected profit
    obj = gp.LinExpr()
    for n in nodes:
        nid = n['id']
        prob = n['probability']
        price = n['electricity_price_dollars_per_MWh']

        # Revenue from selling electricity
        obj += prob * price * x[nid]
        # Fuel cost (linear: a*y + b*x)
        obj -= prob * fuel_a * y[nid]
        obj -= prob * fuel_b * x[nid]
        # Startup cost
        obj -= prob * SU_cost * u[nid]
        # Shutdown cost
        obj -= prob * SD_cost * v[nid]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    for n in nodes:
        nid = n['id']
        pid = n['parent_id']

        # Generation bounds linked to on/off status
        model.addConstr(x[nid] >= C_lower * y[nid], name=f"gen_lb_{nid}")
        model.addConstr(x[nid] <= C_upper * y[nid], name=f"gen_ub_{nid}")

        if pid is None:
            # Root node: parent state is (y0, x0)
            # Startup logic
            model.addConstr(u[nid] <= y[nid], name=f"u_c1_{nid}")
            model.addConstr(u[nid] <= 1 - y0, name=f"u_c2_{nid}")
            model.addConstr(u[nid] >= y[nid] - y0, name=f"u_c3_{nid}")
            # Shutdown definition: v = y0 - y + u
            model.addConstr(v[nid] == y0 - y[nid] + u[nid], name=f"v_def_{nid}")

            # Ramp constraints from initial state
            model.addConstr(x[nid] - x0 <= V_plus + C_upper * (1 - y0), name=f"ramp_up_{nid}")
            model.addConstr(x0 - x[nid] <= V_minus + C_upper * (1 - y[nid]), name=f"ramp_down_{nid}")
        else:
            # Non-root node
            # Startup logic: u=1 iff y=1 and y_parent=0
            model.addConstr(u[nid] <= y[nid], name=f"u_c1_{nid}")
            model.addConstr(u[nid] <= 1 - y[pid], name=f"u_c2_{nid}")
            model.addConstr(u[nid] >= y[nid] - y[pid], name=f"u_c3_{nid}")
            # Shutdown definition: v = y_parent - y + u
            model.addConstr(v[nid] == y[pid] - y[nid] + u[nid], name=f"v_def_{nid}")

            # Ramp up: limited when parent is on; free jump when parent is off
            model.addConstr(x[nid] - x[pid] <= V_plus + C_upper * (1 - y[pid]), name=f"ramp_up_{nid}")
            # Ramp down: limited when current is on; free drop when shutting down
            model.addConstr(x[pid] - x[nid] <= V_minus + C_upper * (1 - y[nid]), name=f"ramp_down_{nid}")

        # Minimum up time: if startup at nid, all descendants within L_up-1 steps must be on
        if L_up > 1 and nid in desc_up:
            for did in desc_up[nid]:
                model.addConstr(y[did] >= u[nid], name=f"min_up_{nid}_{did}")

        # Minimum down time: if shutdown at nid, all descendants within L_down-1 steps must be off
        if L_down > 1 and nid in desc_down:
            for did in desc_down[nid]:
                model.addConstr(y[did] <= 1 - v[nid], name=f"min_down_{nid}_{did}")

    # Callback to log intermediate solutions
    best_logged_obj = -float('inf')
    non_root_ids = set(n['id'] for n in nodes if n['parent_id'] is not None)

    def mip_callback(model, where):
        nonlocal best_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_logged_obj + 1e-6:
                best_logged_obj = obj_val
                x_sol = {}
                y_sol = {}
                u_sol = {}
                for n in nodes:
                    nid = n['id']
                    xv = model.cbGetSolution(x[nid])
                    yv = model.cbGetSolution(y[nid])
                    uv = model.cbGetSolution(u[nid])
                    if xv is not None and xv > 1e-6:
                        x_sol[str(nid)] = round(xv, 6)
                    if yv is not None and yv > 0.5:
                        y_sol[str(nid)] = 1
                    if uv is not None and uv > 0.5 and nid in non_root_ids:
                        u_sol[str(nid)] = 1

                sol = {
                    "objective_value": obj_val,
                    "x": x_sol,
                    "y": y_sol,
                    "u": u_sol,
                    "sparse_defaults": {"x": 0, "y": 0, "u": 0}
                }
                if logger:
                    logger.log_solution(obj_val, sol)

    model.optimize(mip_callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal

        x_sol = {}
        y_sol = {}
        u_sol = {}

        for n in nodes:
            nid = n['id']
            xv = x[nid].X
            yv = y[nid].X
            uv = u[nid].X
            if xv > 1e-6:
                x_sol[str(nid)] = round(xv, 6)
            if yv > 0.5:
                y_sol[str(nid)] = 1
            if uv > 0.5 and nid in non_root_ids:
                u_sol[str(nid)] = 1

        solution = {
            "objective_value": obj_val,
            "x": x_sol,
            "y": y_sol,
            "u": u_sol,
            "sparse_defaults": {"x": 0, "y": 0, "u": 0}
        }

        if logger and obj_val > best_logged_obj + 1e-6:
            logger.log_solution(obj_val, solution)
    else:
        # Fallback: all off (always feasible)
        solution = {
            "objective_value": 0.0,
            "x": {},
            "y": {},
            "u": {},
            "sparse_defaults": {"x": 0, "y": 0, "u": 0}
        }
        if logger:
            logger.log_solution(0.0, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()